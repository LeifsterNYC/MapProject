"""Local Geofabrik PBF extracts as the primary map-data source.

Drop region files (e.g. us-northeast-latest.osm.pbf, norway-latest.osm.pbf)
into ./pbf/ — downloads at https://download.geofabrik.de. Each file's header
declares its coverage bbox; any request whose area a local file covers is
built entirely offline, with Overpass as the fallback for uncovered areas.
No rate limits, no 20-minute fetches, no network flakiness.
"""
import os
import shutil
import subprocess
import tempfile
import warnings

PBF_DIR = os.environ.get(
    'RAMBLE_PBF_DIR', os.path.join(os.path.dirname(__file__), 'pbf'))

# osmium (C++) pre-clips the region extract to the request bbox in seconds;
# pyrosm then parses only the small clip instead of re-scanning the full
# 1.5+ GB file per call (which took 5-10 minutes each). Cached per process
# so the connectivity loop and the feature pass reuse the same clip.
_CLIP_CACHE = {}

# Tag families Ramble reads (network_graph + features). Everything else in
# the clip — buildings, addresses, POIs, the bulk of any metro area — is
# dead weight that pyrosm would still scan single-threaded: an unfiltered
# Ithaca->NYC corridor clip took 2+ hours in _get_pbf_elements alone.
# osmium tags-filter keeps matched objects with ALL their tags plus their
# referenced nodes, so maxspeed/ref/lanes/scenic on highway ways survive.
_KEEP_TAGS = ['highway', 'natural', 'waterway', 'landuse', 'leisure', 'tourism']


def _clipped(path, bbox):
    if not shutil.which('osmium'):
        return path  # slow fallback: pyrosm scans the full file
    key = (path, tuple(round(v, 4) for v in bbox))
    if key not in _CLIP_CACHE:
        w, s, e, n = bbox
        tmp = tempfile.NamedTemporaryFile(suffix='.osm.pbf', delete=False)
        tmp.close()
        subprocess.run(
            ['osmium', 'extract', '--overwrite', '-b', f'{w},{s},{e},{n}',
             '-s', 'complete_ways', '-o', tmp.name, path],
            check=True, capture_output=True)
        filtered = tempfile.NamedTemporaryFile(suffix='.osm.pbf', delete=False)
        filtered.close()
        subprocess.run(
            ['osmium', 'tags-filter', '--overwrite', '-o', filtered.name,
             tmp.name] + _KEEP_TAGS,
            check=True, capture_output=True)
        os.remove(tmp.name)
        _CLIP_CACHE[key] = filtered.name
    return _CLIP_CACHE[key]

_SHORT_CLASSES = [
    'motorway', 'trunk', 'primary', 'secondary', 'tertiary', 'unclassified',
    'residential', 'service', 'motorway_link', 'trunk_link', 'primary_link',
    'secondary_link', 'tertiary_link',
]
_LONG_CLASSES = [
    'motorway', 'trunk', 'primary', 'secondary',
    'motorway_link', 'trunk_link', 'primary_link', 'secondary_link',
]


def find_covering_pbf(bbox):
    """Return the path of a local PBF whose header bbox contains (w, s, e, n),
    or None. Unreadable files (e.g. mid-download) are skipped."""
    import osmium
    w, s, e, n = bbox
    if not os.path.isdir(PBF_DIR):
        return None
    for fname in sorted(os.listdir(PBF_DIR)):
        if not fname.endswith('.pbf'):
            continue
        path = os.path.join(PBF_DIR, fname)
        try:
            box = osmium.io.Reader(path).header().box()
        except Exception:
            continue
        if not box.valid():
            continue
        if (box.bottom_left.lon <= w and box.bottom_left.lat <= s and
                box.top_right.lon >= e and box.top_right.lat >= n):
            return path
    return None


def network_graph(path, bbox, long_trip):
    """Build a simplified osmnx-style MultiDiGraph from a local PBF, filtered
    to the same road classes as the Overpass path, all components retained."""
    import osmnx
    from pyrosm import OSM
    w, s, e, n = bbox
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        osm = OSM(_clipped(path, bbox), bounding_box=[w, s, e, n])
        nodes, edges = osm.get_network(nodes=True, network_type='driving')
        # pyrosm can emit duplicate node ids when ways cross the bounding box;
        # to_graph's osmnx conversion requires a unique node index.
        nodes = nodes.drop_duplicates(subset='id').reset_index(drop=True)
        classes = _LONG_CLASSES if long_trip else _SHORT_CLASSES
        edges = edges[edges['highway'].isin(classes)]
        if 'access' in edges.columns:
            edges = edges[edges['access'].isna() | (edges['access'] != 'private')]
        # to_graph mis-indexes filtered frames with gap-y indexes ("index must
        # be unique") — give it a clean RangeIndex.
        edges = edges.reset_index(drop=True)
        graph = osm.to_graph(nodes, edges, graph_type='networkx',
                             osmnx_compatible=True, retain_all=True)
    simplify = getattr(osmnx, 'simplify_graph', None) or osmnx.simplification.simplify_graph
    return simplify(graph)


_WATER_NATURAL = ['water', 'coastline']
_NATURE_NATURAL = ['wood', 'scrub', 'heath', 'grassland']
_NATURE_LANDUSE = ['forest', 'meadow', 'grass', 'village_green', 'recreation_ground']
_NATURE_LEISURE = ['park', 'nature_reserve']


def features(path, bbox, long_trip):
    """Return feature GeoDataFrame lists from a local PBF, mirroring the
    Overpass feature set: {'water', 'nature', 'viewpoints', 'signals'}.

    ONE combined scan: pyrosm's cost is proportional to file size, not bbox,
    so each get_data_by_custom_criteria call re-reads the whole extract.
    Fetch everything in a single pass and split by tag afterwards.
    """
    from pyrosm import OSM
    w, s, e, n = bbox
    if long_trip:
        custom_filter = {
            'natural': ['coastline'],
            'waterway': ['river'],
            'tourism': ['viewpoint', 'scenic_viewpoint'],
        }
    else:
        custom_filter = {
            'natural': _WATER_NATURAL + _NATURE_NATURAL,
            'waterway': ['river', 'stream'],
            'landuse': _NATURE_LANDUSE,
            'leisure': _NATURE_LEISURE,
            'tourism': ['viewpoint', 'scenic_viewpoint'],
            'highway': ['traffic_signals', 'stop'],
        }
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        osm = OSM(_clipped(path, bbox), bounding_box=[w, s, e, n])
        gdf = osm.get_data_by_custom_criteria(
            custom_filter=custom_filter,
            keep_nodes=True, keep_ways=True, keep_relations=True)

    out = {'water': [], 'nature': [], 'viewpoints': [], 'signals': []}
    if gdf is None or not len(gdf):
        return out

    def col(name):
        return gdf[name] if name in gdf.columns else None

    def subset(mask):
        part = gdf[mask]
        return [part] if len(part) else []

    import pandas as pd
    false = pd.Series(False, index=gdf.index)
    natural = col('natural')
    water_mask = false.copy()
    if natural is not None:
        water_mask |= natural.isin(['coastline'] if long_trip else _WATER_NATURAL)
    ww = col('waterway')
    if ww is not None:
        water_mask |= ww.isin(['river'] if long_trip else ['river', 'stream'])
    out['water'] = subset(water_mask)

    tourism = col('tourism')
    if tourism is not None:
        out['viewpoints'] = subset(tourism.isin(['viewpoint', 'scenic_viewpoint']))

    if not long_trip:
        nature_mask = false.copy()
        if natural is not None:
            nature_mask |= natural.isin(_NATURE_NATURAL)
        lu = col('landuse')
        if lu is not None:
            nature_mask |= lu.isin(_NATURE_LANDUSE)
        le = col('leisure')
        if le is not None:
            nature_mask |= le.isin(_NATURE_LEISURE)
        out['nature'] = subset(nature_mask)

        hw = col('highway')
        if hw is not None:
            out['signals'] = subset(hw.isin(['traffic_signals', 'stop']))
    return out
