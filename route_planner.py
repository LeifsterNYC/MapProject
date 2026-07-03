import osmnx
import networkx
import folium
import math
import os
import pickle
import re
import time
import numpy as np
from scipy.spatial import KDTree

try:  # osmnx raises this when a query matches zero elements — a legit result
    from osmnx._errors import InsufficientResponseError as _EmptyResponse
except ImportError:  # fallback if osmnx moves it
    class _EmptyResponse(Exception):
        pass

# Keep the explicit scenic=yes way tag — OSM's direct "this road is scenic"
# signal, common in Europe. Must be set before any graph fetch.
if 'scenic' not in osmnx.settings.useful_tags_way:
    osmnx.settings.useful_tags_way = list(osmnx.settings.useful_tags_way) + ['scenic']

_GRAPH_CACHE: dict = {}
_CACHE_DIR = os.path.join(os.path.dirname(__file__), '.graph_cache')
_CACHE_VERSION = 5  # Bump to invalidate all on-disk caches when fetched data changes.
_SIGNAL_DELAY_HR = 35 / 3600  # 35-second stop penalty at signalized intersections.
_STOP_DELAY_HR = 12 / 3600    # 12-second penalty at stop signs.

DEFAULT_SPEEDS = {
    'motorway': 65, 'trunk': 55, 'primary': 55, 'secondary': 45,
    'tertiary': 40, 'unclassified': 35, 'motorway_link': 45,
    'trunk_link': 40, 'primary_link': 35, 'secondary_link': 35,
    'tertiary_link': 30, 'residential': 35, 'living_street': 15,
    'service': 10, 'pedestrian': 5,
}


# Trips longer than this (straight-line miles) switch to long-trip mode:
# a highway-class-only network over the bbox, river + viewpoint features
# only, no signal fetch. At that scale the scenic decision is which
# corridors to take (NY-17 vs I-380/80, Rt 97 vs I-84), not side streets.
LONG_TRIP_MILES = 45

_SHORT_FILTER = '["area"!~"yes"]["highway"~"motorway|trunk|primary|secondary|tertiary|unclassified|residential|service|motorway_link|trunk_link|primary_link|secondary_link|tertiary_link"]["access"!~"private"]'
_LONG_FILTER = '["area"!~"yes"]["highway"~"motorway|trunk|primary|secondary|motorway_link|trunk_link|primary_link|secondary_link"]["access"!~"private"]'


def _parse_maxspeed_mph(raw):
    """Parse an OSM maxspeed tag into mph.

    OSM's default unit is km/h worldwide; only explicit '<n> mph' tags (US/UK)
    are miles. Treating bare numbers as mph made every non-US travel time
    ~1.6x too optimistic. Returns None for unparseable values ('none',
    'signals', missing).
    """
    if isinstance(raw, list):
        raw = raw[0] if raw else None
    if isinstance(raw, (int, float)):
        return float(raw) * 0.621371
    if isinstance(raw, str):
        try:
            value = float(raw.split()[0])
        except (ValueError, IndexError):
            return None
        return value if 'mph' in raw.lower() else value * 0.621371
    return None


def _straight_miles(start, end):
    """Haversine distance in miles between two (lat, lon) points."""
    lat1, lon1 = start
    lat2, lon2 = end
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    h = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlam / 2) ** 2
    return 2 * 3958.8 * math.asin(math.sqrt(h))


def _fetch_features(bbox, tags, retries=2, pause_s=20):
    """Fetch OSM features with retry + per-tag-key fallback.

    Overpass sheds heavy queries under load (connection refused) while small
    ones pass, so after retries the combined query is split per tag key.
    Returns (list_of_gdfs, ok). A query matching zero elements is a legitimate
    empty result (ok=True), not a failure.
    """
    for attempt in range(retries):
        try:
            return [osmnx.features_from_bbox(bbox, tags=tags)], True
        except _EmptyResponse:
            return [], True
        except Exception:
            time.sleep(pause_s * (attempt + 1))
    gdfs, ok = [], True
    for key, val in tags.items():
        try:
            gdfs.append(osmnx.features_from_bbox(bbox, tags={key: val}))
        except _EmptyResponse:
            continue
        except Exception:
            ok = False
    return gdfs, ok


def _polygon_points(gdfs, subsample=4):
    """Extract (lat, lon) vertices from feature geometries (any gdf list)."""
    points = []
    for gdf in gdfs:
        for geom in gdf.geometry:
            if geom is None:
                continue
            try:
                if hasattr(geom, 'exterior'):
                    geom_coords = list(geom.exterior.coords)
                elif hasattr(geom, 'coords'):
                    geom_coords = list(geom.coords)
                elif hasattr(geom, 'geoms'):
                    geom_coords = []
                    for part in geom.geoms:
                        if hasattr(part, 'exterior'):
                            geom_coords.extend(part.exterior.coords)
                        elif hasattr(part, 'coords'):
                            geom_coords.extend(part.coords)
                else:
                    geom_coords = []
                points.extend((y, x) for x, y in geom_coords[::subsample])
            except Exception:
                continue
    return points


def initialize_graph(start, end):
    buffer = 0.1
    n = max(start[0], end[0]) + buffer
    s = min(start[0], end[0]) - buffer
    e = max(start[1], end[1]) + buffer
    w = min(start[1], end[1]) - buffer
    long_trip = _straight_miles(start, end) > LONG_TRIP_MILES

    # Round to 2 decimal places (~1 km) so nearby routes share a cache entry.
    bbox_key = (round(n, 2), round(s, 2), round(e, 2), round(w, 2), long_trip)

    if bbox_key not in _GRAPH_CACHE:
        prefix = 'long_' if long_trip else ''
        cache_file = os.path.join(_CACHE_DIR, f'v{_CACHE_VERSION}_{prefix}graph_{bbox_key[:4]}.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                graph = pickle.load(f)
        else:
            filt = _LONG_FILTER if long_trip else _SHORT_FILTER
            graph = osmnx.graph_from_bbox((w, s, e, n), simplify=True, network_type='drive', custom_filter=filt)

            for u, v, k, data in graph.edges(data=True, keys=True):
                highway = data.get('highway', 'residential')
                if isinstance(highway, list):
                    highway = highway[0]
                speed = _parse_maxspeed_mph(data.get('maxspeed'))
                if not speed or speed <= 0:
                    speed = DEFAULT_SPEEDS.get(highway, 25)
                data['maxspeed'] = speed
                data['travel_time'] = data['length'] / 1609.34 / speed

            # Penalize signalized intersections and stop signs by PROXIMITY, not
            # node tags: OSM tags signals on pole/approach nodes that graph
            # simplification folds into edge geometry, so tag-based matching
            # missed nearly all of them (measured: 1 of ~10 lights counted on a
            # cross-town route). Fetch signal/stop points as features, cluster
            # poles into intersections, and penalize each edge whose end node
            # lands near one — one penalty per intersection crossing.
            # A failed feature fetch (Overpass rate-limiting) must not get
            # pickled: the graph would be cached forever with silently-empty
            # water/viewpoints. Serve it from memory but skip the disk write.
            features_ok = True

            # Long-trip graphs are highway-class only: skip the signal fetch
            # (rarely signalized, and the point fetch at that scale is huge).
            signal_pts, stop_pts = [], []
            if not long_trip:
                gdfs, ok = _fetch_features((w, s, e, n), {'highway': ['traffic_signals', 'stop']})
                features_ok = features_ok and ok
                for sig_gdf in gdfs:
                    pts = sig_gdf.geometry.representative_point()
                    kinds = sig_gdf.get('highway')
                    signal_pts += [(y, x) for y, x, kind in zip(pts.y, pts.x, kinds)
                                   if kind == 'traffic_signals']
                    stop_pts += [(y, x) for y, x, kind in zip(pts.y, pts.x, kinds)
                                 if kind == 'stop']

            def _cluster(points, cell_deg=0.0005):
                """Merge points within ~50 m so multi-pole intersections count once."""
                cells = {}
                for lat, lon in points:
                    cells.setdefault((round(lat / cell_deg), round(lon / cell_deg)), (lat, lon))
                return list(cells.values())

            signal_clusters = _cluster(signal_pts)
            stop_clusters = _cluster(stop_pts)
            graph.graph['signal_points'] = signal_clusters

            mid_lat = (n + s) / 2
            lat_m = 111320.0
            lon_m = 111320.0 * math.cos(math.radians(mid_lat))
            sig_tree = (KDTree(np.array([(la * lat_m, lo * lon_m) for la, lo in signal_clusters]))
                        if signal_clusters else None)
            stop_tree = (KDTree(np.array([(la * lat_m, lo * lon_m) for la, lo in stop_clusters]))
                         if stop_clusters else None)
            for u, v, k, data in graph.edges(data=True, keys=True):
                vy, vx = graph.nodes[v]['y'], graph.nodes[v]['x']
                q = [vy * lat_m, vx * lon_m]
                if sig_tree is not None and sig_tree.query(q)[0] < 40:
                    data['travel_time'] += _SIGNAL_DELAY_HR
                elif stop_tree is not None and stop_tree.query(q)[0] < 25:
                    data['travel_time'] += _STOP_DELAY_HR

            # Fetch water bodies for proximity scoring.
            # Collect shoreline/centerline vertices — representative_point lands in the
            # middle of wide rivers (too far from shore for a 400m proximity check).
            # Coastline is included in both modes: fjords and sea shores are
            # natural=coastline in OSM, not natural=water — without it a road
            # hugging Geirangerfjord or Vestfjorden gets no water credit. Long
            # trips skip lakes/streams (enormous at corridor scale); river
            # valleys + coasts are the signal that matters.
            water_tags = ({'waterway': 'river', 'natural': 'coastline'} if long_trip else
                          {'natural': ['water', 'coastline'], 'waterway': ['river', 'stream']})
            gdfs, ok = _fetch_features((w, s, e, n), water_tags)
            features_ok = features_ok and ok
            graph.graph['water_points'] = _polygon_points(gdfs)

            # Fetch roadside viewpoints (tourism=viewpoint only — not natural=peak,
            # which are tagged on mountain tops far from roads).
            gdfs, ok = _fetch_features((w, s, e, n), {'tourism': ['viewpoint', 'scenic_viewpoint']})
            features_ok = features_ok and ok
            viewpoints = []
            for vp_gdf in gdfs:
                pts = vp_gdf.geometry.representative_point()
                viewpoints += list(zip(pts.y, pts.x))
            graph.graph['viewpoints'] = viewpoints

            # Fetch natural areas (forests, parks, meadows) for proximity scoring.
            # Skipped on long trips: a percentile nature signal over a
            # 25,000 km² corridor separates nothing, and the fetch is huge.
            nature_points = []
            if not long_trip:
                gdfs, ok = _fetch_features((w, s, e, n), {
                    'natural': ['wood', 'scrub', 'heath', 'grassland'],
                    'landuse': ['forest', 'meadow', 'grass', 'village_green', 'recreation_ground'],
                    'leisure': ['park', 'nature_reserve'],
                })
                features_ok = features_ok and ok
                nature_points = _polygon_points(gdfs)
            graph.graph['nature_points'] = nature_points

            if features_ok:
                os.makedirs(_CACHE_DIR, exist_ok=True)
                with open(cache_file, 'wb') as f:
                    pickle.dump(graph, f)
            else:
                print(f"WARNING: feature fetch failed for {bbox_key} — "
                      f"graph served from memory, NOT cached (retry later)")
        _GRAPH_CACHE[bbox_key] = graph

    graph = _GRAPH_CACHE[bbox_key]
    midpoint_lat = (n + s) / 2
    midpoint_lon = (e + w) / 2
    route_map = folium.Map(location=[midpoint_lat, midpoint_lon], zoom_start=10)
    return graph, route_map


# Full-score curviness: accumulated bearing change per metre of road.
# 0.30 deg/m = 30 degrees per 100 m sustained — a genuinely twisty road.
_CURVE_FULL_SCORE_DEG_PER_M = 0.30
# Segments shorter than this are merged before computing bearings, and
# deflections smaller than the noise floor are ignored: OSM geometry is
# hand-digitized and tiny vertex jitter would otherwise accumulate.
_CURVE_MIN_SEG_M = 15.0
_CURVE_NOISE_DEG = 5.0


def _curviness_score(data):
    """Bearing-change density along the edge geometry, scaled to [0, 1].

    Chord ratio (length/straight - 1) is blind to ordinary winding roads:
    OSMnx edges end at intersections, and a road that snakes but progresses
    scores ~0 (measured 0.006-0.03 across whole graphs). Summing heading
    changes measures what a driver feels.
    """
    geom = data.get('geometry')
    length = data.get('length', 0)
    if geom is None or length < 3 * _CURVE_MIN_SEG_M:
        return 0.0
    coords = list(geom.coords)
    if len(coords) < 3:
        return 0.0
    lat_ref = coords[0][1]
    lon_m = 111320.0 * math.cos(math.radians(lat_ref))
    lat_m = 111320.0

    # Thin vertices so every retained segment is at least _CURVE_MIN_SEG_M.
    pts = [(coords[0][0] * lon_m, coords[0][1] * lat_m)]
    for x, y in coords[1:]:
        px, py = x * lon_m, y * lat_m
        lx, ly = pts[-1]
        if math.hypot(px - lx, py - ly) >= _CURVE_MIN_SEG_M:
            pts.append((px, py))
    if len(pts) < 3:
        return 0.0

    total_deg = 0.0
    prev_bearing = None
    for i in range(1, len(pts)):
        dx = pts[i][0] - pts[i - 1][0]
        dy = pts[i][1] - pts[i - 1][1]
        bearing = math.degrees(math.atan2(dy, dx))
        if prev_bearing is not None:
            turn = abs(bearing - prev_bearing)
            if turn > 180.0:
                turn = 360.0 - turn
            if turn > _CURVE_NOISE_DEG:
                total_deg += turn
        prev_bearing = bearing

    return min(total_deg / length / _CURVE_FULL_SCORE_DEG_PER_M, 1.0)


# Numbered-highway refs that mark pleasant country/state routes: US state
# style (NY 97, US 44, SR 9) and Norwegian riks-/fylkesveier (Rv 15, Fv 815).
# Word boundaries prevent false hits (e.g. 'NE 2' must not match 'E').
_SCENIC_REF_RE = re.compile(r'\b(NY|US|SR|Rv|Fv)\s?-?\d', re.IGNORECASE)


def _road_type_score(data):
    highway = data.get('highway', 'residential')
    if isinstance(highway, list):
        highway = highway[0]
    table = {
        'motorway': 0.15, 'trunk': 0.15, 'primary': 0.3,
        'motorway_link': 0.0, 'trunk_link': 0.0, 'primary_link': 0.1,
        'secondary': 0.5, 'secondary_link': 0.5,
        'tertiary': 0.9, 'tertiary_link': 0.9,
        'unclassified': 0.7, 'residential': 0.3,
        'living_street': 0.5, 'service': 0.05,
    }
    score = table.get(highway, 0.4)
    # OSM's explicit scenic tag wins outright — any road class (the Palisades
    # Parkway is motorway-class and genuinely scenic).
    scenic = data.get('scenic')
    if isinstance(scenic, list):
        scenic = scenic[0]
    if scenic == 'yes':
        return max(score, 0.9)
    # Scenic numbered highways carry a state/county route ref but OSM class
    # tags that score them like traffic arteries or driveways (NY-44/55 and
    # NY-97 are 'primary' → 0.3; some are even 'residential'). A ref rescue
    # restores them to country-highway level. Trunk/motorway stay excluded,
    # and so are European E-refs — "E 10" is a trunk network designation, not
    # a scenic road (in Lofoten it is precisely the road to avoid).
    if highway in ('residential', 'unclassified', 'tertiary',
                   'primary', 'secondary', 'primary_link', 'secondary_link'):
        ref = data.get('ref')
        if isinstance(ref, list):
            ref = ref[0]
        # Bare numeric refs are numbered county/provincial roads in much of
        # Europe (Norwegian fylkesvei tag ref=815, not 'Fv 815').
        if isinstance(ref, str) and (_SCENIC_REF_RE.search(ref) or ref.strip().isdigit()):
            score = max(score, 0.8)
    return score



def _traffic_score(data):
    lanes = data.get('lanes')
    if lanes is None:
        return 0.3  # no lanes tag → probably quiet, but do not over-reward
    try:
        n = int(lanes) if not isinstance(lanes, list) else int(lanes[0])
    except (ValueError, TypeError):
        return 0.2
    if n <= 1:
        return 0.5
    elif n == 2:
        return 0.2
    else:
        return 0.0  # 3+ lanes = busy road


def _speed_score(data):
    speed = data.get('maxspeed', 35)
    # Peaks at 45-55 mph (open country road).
    if speed >= 65:
        return 0.4
    elif speed >= 55:
        return 1.0
    elif speed >= 45:
        return 0.8
    elif speed >= 35:
        return 0.5
    else:
        return 0.2


def _make_tree(points, lat_m, lon_m):
    """Project (lat, lon) points to approximate metres and return a KDTree."""
    if not points:
        return None
    arr = np.array([(lat * lat_m, lon * lon_m) for lat, lon in points])
    return KDTree(arr)


def relative_proximity_scores(distances, far_gate_m=2000.0):
    """Convert per-edge distances-to-a-feature into a discriminating [0, 1] score.

    Percentile-rank based: the closest edge scores 1.0, the farthest 0.0, spread
    evenly in between — so the signal separates edges *within this graph* instead
    of saturating near 1.0 in feature-rich regions (the root-cause bug).

    If the feature is effectively absent (median distance beyond far_gate_m, or an
    empty/all-infinite distance array), every edge scores 0.0 so a missing feature
    contributes nothing rather than rewarding merely-least-far edges.
    """
    d = np.asarray(distances, dtype=float)
    n = d.size
    if n == 0:
        return d
    finite = d[np.isfinite(d)]
    if finite.size == 0 or np.median(finite) > far_gate_m:
        return np.zeros(n)
    order = d.argsort()
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.arange(n)
    return 1.0 - ranks / max(n - 1, 1)


def score_scenic_edges(G, weights: dict) -> None:
    w_curve = weights.get('curviness', 1.0)
    w_road = weights.get('road_type', 1.0)
    w_nature = weights.get('nature', 1.0)
    # Fixed component weights: speed, traffic, water, viewpoints. Curviness is
    # internally boosted (it is the one signal that identifies famous mountain
    # roads); dense percentile water/nature are damped so they cannot drown it;
    # viewpoints get a wide 600 m band so overlook roads actually earn credit.
    W_SPEED, W_TRAFFIC, W_WATER, W_VIEW = 1.0, 0.5, 0.5, 2.0
    CURVE_BOOST, NATURE_DAMP = 2.5, 0.5
    VIEW_BAND_M = 600.0
    # Viewpoint-cluster bonus: a road with several overlooks nearby (the 44/55
    # ridge has up to 7 within 1 km; ordinary roads have 0) is a destination
    # drive. Threshold >= 3 qualifies only 1.7% of edges in viewpoint-rich
    # graphs and none elsewhere — surgical, no wandering collateral.
    W_VIEWCNT, VCNT_BAND_M, VCNT_THRESH = 3.0, 1000.0, 3

    # Project to metres using a single reference latitude (error < 0.5% over 50-mile bbox).
    ref_lat = sum(d.get('y', 0) for _, d in list(G.nodes(data=True))[:200]) / 200
    lat_m = 111320.0
    lon_m = 111320.0 * math.cos(math.radians(ref_lat))

    water_tree = _make_tree(G.graph.get('water_points', []), lat_m, lon_m)
    view_tree = _make_tree(G.graph.get('viewpoints', []), lat_m, lon_m)
    nature_tree = _make_tree(G.graph.get('nature_points', []), lat_m, lon_m)

    edges = list(G.edges(data=True, keys=True))
    mids = []
    for u, v, k, data in edges:
        geom = data.get('geometry')
        if geom is not None:
            coords = list(geom.coords)
            mid = coords[len(coords) // 2]
            mids.append((mid[1], mid[0]))
        else:
            mids.append(((G.nodes[u]['y'] + G.nodes[v]['y']) / 2,
                         (G.nodes[u]['x'] + G.nodes[v]['x']) / 2))
    mids_m = np.array([(lat * lat_m, lon * lon_m) for lat, lon in mids])

    def dists_to(tree):
        if tree is None:
            return np.full(len(mids), np.inf)
        dist, _ = tree.query(mids_m)
        return dist

    # Dense features (water, nature: thousands of points in a green region) use
    # relative (percentile-rank) proximity so they discriminate within this graph
    # instead of saturating near 1.0 everywhere. Sparse viewpoints keep an
    # absolute 200 m gradient — ranking would zero them out (the median edge is
    # legitimately far from any viewpoint).
    water_rel = relative_proximity_scores(dists_to(water_tree))
    nature_rel = relative_proximity_scores(dists_to(nature_tree))
    view_abs = np.maximum(0.0, 1.0 - dists_to(view_tree) / VIEW_BAND_M)
    if view_tree is not None:
        vcount = np.asarray(view_tree.query_ball_point(mids_m, r=VCNT_BAND_M, return_length=True))
    else:
        vcount = np.zeros(len(mids))
    vcnt_term = (vcount >= VCNT_THRESH).astype(float)

    # Urban anti-wandering gate: in dense street grids the water/nature/curviness
    # bonuses reward pointless zigzagging through city blocks (measured on the
    # Ithaca grid, where the blocks are OSM-tagged 'unclassified', so tag-based
    # gating misses them). Damp those bonuses by intersection density instead:
    # full credit below ~25 graph nodes within 300 m (rural roads measure <=25),
    # ramping down to 0.2 credit at >=60 (downtown grids measure 33-140).
    node_arr = np.array([(d['y'] * lat_m, d['x'] * lon_m) for _, d in G.nodes(data=True)])
    node_tree = KDTree(node_arr)
    density = node_tree.query_ball_point(mids_m, r=300.0, return_length=True)
    gate = np.clip(1.0 - 0.8 * (density - 25.0) / 35.0, 0.2, 1.0)

    scores = np.empty(len(edges))
    for i, (u, v, k, data) in enumerate(edges):
        g = float(gate[i])
        road = _road_type_score(data)
        # Curviness only counts on roads worth driving: curvy driveways,
        # cul-de-sacs and parking loops saturate the bearing metric (21% of
        # edges score >0.3) and must not become scenic attractors.
        curve = _curviness_score(data) if road >= 0.5 else 0.0
        scores[i] = (
            g * (
                CURVE_BOOST * w_curve * curve +
                NATURE_DAMP * w_nature * float(nature_rel[i]) +
                W_WATER * float(water_rel[i])
            ) +
            w_road * road +
            W_SPEED * _speed_score(data) +
            W_TRAFFIC * _traffic_score(data) +
            W_VIEW * float(view_abs[i]) +
            W_VIEWCNT * float(vcnt_term[i])
        )

    # Empirical normalization: percentile-rank the total score across the graph.
    # Summed component scores cluster in a narrow band (measured ~1.5-2.3 of a
    # theoretical 6.0), so dividing by the theoretical max leaves the routing
    # signal too compressed to matter. Ranking spreads scenic_t over [0, 1].
    order = scores.argsort()
    ranks = np.empty(len(edges))
    ranks[order] = np.arange(len(edges))
    t_all = ranks / max(len(edges) - 1, 1)

    for i, (u, v, k, data) in enumerate(edges):
        data['scenic_score'] = float(scores[i])
        data['scenic_t'] = float(t_all[i])


# K values probed by plan_scenic_route. Route choice is knife-edged in K (a
# single fixed K yields either the fast route or a blown detour budget depending
# on the trip), so we probe a ladder and pick the best route within budget.
_K_LADDER = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0)

# Detour budget: additive, ceilinged at 12 extra minutes (or 20% of the fast
# time on long trips, whichever is larger). A multiplicative cap is wrong at
# both ends — too tight on a 5-minute trip and so loose on a 20+ minute trip
# that 19-minute twisty excursions slip in. Calibration showed no 4/4 window
# exists with a multiplicative cap. The 20% term lets a 4-hour trip earn a
# ~45-minute scenic corridor swap (17/97/Palisades vs 380/80).
_DETOUR_SLACK_HR = 12 / 60
_DETOUR_SLACK_FRAC = 0.20

# Selection objective: scenic gain minus a flat per-minute time penalty.
# Calibrated by grid search over the fixture candidate tables with live
# curviness + the viewpoint-cluster term + the additive budget: the 4/4
# window is (0, 0.0264], and 0.013 is its center. Re-run `python evaluate.py`
# after ANY scoring change.
_TIME_PENALTY_PER_MIN = 0.013  # scenic-score units per extra minute


def _trim_to_simple(route):
    """Collapse repeated-node cycles so an out-and-back anchor leg becomes a
    simple path. The edge leaving a node's last occurrence follows its first
    occurrence, so edge continuity is preserved."""
    out = []
    pos = {}
    for node in route:
        if node in pos:
            cut = pos[node] + 1
            for n in out[cut:]:
                del pos[n]
            out = out[:cut]
        else:
            pos[node] = len(out)
            out.append(node)
    return out


def _scenic_anchors(G, max_anchors=4):
    """Pick up to max_anchors nodes in the most scenic clusters of the graph.

    The K-ladder cannot generate a distant geographic excursion — a cheaper
    connected in-town path always dominates under global exp reweighting
    (verified: no K in [1, 20] ever surfaces a known scenic loop). Seeding
    candidates through top-scenic clusters fixes that.
    """
    best = []  # (scenic_t, node, lat, lon) for top-t edge endpoints
    for u, v, k, data in G.edges(data=True, keys=True):
        t = data.get('scenic_t', 0.0)
        if t >= 0.99:
            best.append((t, v, G.nodes[v]['y'], G.nodes[v]['x']))
    if not best:
        return []
    # Grid-bin (~2 km cells) and rank cells by summed scenic_t.
    cells = {}
    for t, node, lat, lon in best:
        key = (round(lat / 0.02), round(lon / 0.02))
        weight, _ = cells.get(key, (0.0, node))
        cells[key] = (weight + t, node)
    ranked = sorted(cells.items(), key=lambda kv: kv[1][0], reverse=True)
    anchors = []
    taken = []
    for (cy, cx), (weight, node) in ranked:
        if any(abs(cy - ty) <= 1 and abs(cx - tx) <= 1 for ty, tx in taken):
            continue  # skip cells adjacent to an already-chosen cluster
        anchors.append(node)
        taken.append((cy, cx))
        if len(anchors) >= max_anchors:
            break
    return anchors


def _tw_mean_score(G, path):
    """Travel-time-weighted mean scenic_score along a path."""
    total = 0.0
    weighted = 0.0
    for u, v in zip(path[:-1], path[1:]):
        data = min(G[u][v].values(), key=lambda d: d.get('travel_time', 0))
        tt = data.get('travel_time', 0.0)
        weighted += data.get('scenic_score', 0.0) * tt
        total += tt
    return weighted / total if total else 0.0


def plan_scenic_route(G, start, end, max_detour_factor) -> tuple:
    """Returns (fast_route, scenic_route, fast_minutes, scenic_minutes).

    Adaptive detour search: generate candidates from the K-ladder (scenic_cost =
    travel_time * exp(K * (1 - scenic_t)); one Dijkstra each) plus routes seeded
    through top-scenic anchor clusters. Keep candidates within the detour budget
    (multiplicative cap or +15 min, whichever is larger) and return the one with
    the best scenic gain minus linear time penalty. The fast route is the floor
    candidate, so when no detour is worth it the fast route wins naturally.
    """
    fast_route = networkx.shortest_path(G, start, end, weight='travel_time')
    fast_time = networkx.path_weight(G, fast_route, weight='travel_time')
    fast_score = _tw_mean_score(G, fast_route)
    # The max_detour slider can tighten the budget below the ceiling
    # (matters on short trips) but never extend it beyond.
    slack = max(_DETOUR_SLACK_HR, _DETOUR_SLACK_FRAC * fast_time)
    budget = fast_time + min(fast_time * (max_detour_factor - 1.0), slack)

    candidates = []
    seen = {tuple(fast_route)}

    def consider(route):
        key = tuple(route)
        if key in seen:
            return
        seen.add(key)
        candidates.append(route)

    for K in _K_LADDER:
        for u, v, k, data in G.edges(data=True, keys=True):
            data['scenic_cost'] = data['travel_time'] * math.exp(K * (1.0 - data.get('scenic_t', 0.0)))
        try:
            consider(networkx.shortest_path(G, start, end, weight='scenic_cost'))
        except (networkx.NetworkXNoPath, networkx.NodeNotFound):
            break

    # Anchor-seeded candidates: fastest and moderately-scenic (K=3 costs, still
    # set from the ladder's last iteration order — recompute for K=3) legs
    # through each top-scenic cluster.
    for u, v, k, data in G.edges(data=True, keys=True):
        data['scenic_cost'] = data['travel_time'] * math.exp(3.0 * (1.0 - data.get('scenic_t', 0.0)))
    for anchor in _scenic_anchors(G):
        for weight in ('travel_time', 'scenic_cost'):
            try:
                leg_in = networkx.shortest_path(G, start, anchor, weight=weight)
                leg_out = networkx.shortest_path(G, anchor, end, weight=weight)
            except (networkx.NetworkXNoPath, networkx.NodeNotFound):
                continue
            # Trim out-and-back cycles rather than discarding: the trimmed
            # route often keeps the scenic corridor the anchor pulled it
            # through while dropping the pointless spur to the anchor itself.
            consider(_trim_to_simple(leg_in + leg_out[1:]))

    best_route, best_time, best_value = fast_route, fast_time, 0.0
    for route in candidates:
        time = networkx.path_weight(G, route, weight='travel_time')
        if time > budget:
            continue
        extra_min = (time - fast_time) * 60.0
        value = (_tw_mean_score(G, route) - fast_score) - _TIME_PENALTY_PER_MIN * extra_min
        if value > best_value:
            best_route, best_time, best_value = route, time, value

    return fast_route, best_route, fast_time * 60, best_time * 60


def route_coords(G, route):
    """Extract (lat, lon) coords following actual road geometry, not just nodes."""
    coords = []
    for i in range(len(route) - 1):
        u, v = route[i], route[i + 1]
        edge_data = min(G[u][v].values(), key=lambda d: d.get('travel_time', 0))
        geom = edge_data.get('geometry')
        if geom is not None:
            edge_coords = [(y, x) for x, y in geom.coords]
        else:
            edge_coords = [(G.nodes[u]['y'], G.nodes[u]['x']),
                           (G.nodes[v]['y'], G.nodes[v]['x'])]
        coords.extend(edge_coords if not coords else edge_coords[1:])
    if not coords and route:
        coords.append((G.nodes[route[0]]['y'], G.nodes[route[0]]['x']))
    return coords
