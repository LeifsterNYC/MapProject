import osmnx
import networkx
import folium
import math
import os
import pickle
import numpy as np
from scipy.spatial import KDTree

_GRAPH_CACHE: dict = {}
_CACHE_DIR = os.path.join(os.path.dirname(__file__), '.graph_cache')
_CACHE_VERSION = 4  # Bump to invalidate all on-disk caches when fetched data changes.
_SIGNAL_DELAY_HR = 35 / 3600  # 35-second stop penalty at signalized intersections.

DEFAULT_SPEEDS = {
    'motorway': 65, 'trunk': 55, 'primary': 55, 'secondary': 45,
    'tertiary': 40, 'unclassified': 35, 'motorway_link': 45,
    'trunk_link': 40, 'primary_link': 35, 'secondary_link': 35,
    'tertiary_link': 30, 'residential': 35, 'living_street': 15,
    'service': 10, 'pedestrian': 5,
}


def initialize_graph(start, end):
    buffer = 0.1
    n = max(start[0], end[0]) + buffer
    s = min(start[0], end[0]) - buffer
    e = max(start[1], end[1]) + buffer
    w = min(start[1], end[1]) - buffer

    # Round to 2 decimal places (~1 km) so nearby routes share a cache entry.
    bbox_key = (round(n, 2), round(s, 2), round(e, 2), round(w, 2))

    if bbox_key not in _GRAPH_CACHE:
        cache_file = os.path.join(_CACHE_DIR, f'v{_CACHE_VERSION}_graph_{bbox_key}.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                graph = pickle.load(f)
        else:
            private_filter = '["area"!~"yes"]["highway"~"motorway|trunk|primary|secondary|tertiary|unclassified|residential|service|motorway_link|trunk_link|primary_link|secondary_link|tertiary_link"]["access"!~"private"]'
            graph = osmnx.graph_from_bbox((w, s, e, n), simplify=True, network_type='drive', custom_filter=private_filter)

            for u, v, k, data in graph.edges(data=True, keys=True):
                highway = data.get('highway', 'residential')
                if isinstance(highway, list):
                    highway = highway[0]
                raw = data.get('maxspeed')
                speed = None
                if isinstance(raw, str):
                    try:
                        speed = float(raw.split()[0])
                    except (ValueError, IndexError):
                        speed = None
                elif isinstance(raw, (int, float)):
                    speed = float(raw)
                if not speed or speed <= 0:
                    speed = DEFAULT_SPEEDS.get(highway, 25)
                data['maxspeed'] = speed
                data['travel_time'] = data['length'] / 1609.34 / speed

            # Add ~35-second delay at signalized intersections.
            # OSMnx preserves highway=traffic_signals on intersection nodes.
            for node, ndata in graph.nodes(data=True):
                if ndata.get('highway') == 'traffic_signals':
                    for _, _, edata in graph.in_edges(node, data=True):
                        edata['travel_time'] = edata.get('travel_time', 0) + _SIGNAL_DELAY_HR

            # Fetch water bodies for proximity scoring.
            # Collect shoreline/centerline vertices — representative_point lands in the
            # middle of wide rivers (too far from shore for a 400m proximity check).
            try:
                water_gdf = osmnx.features_from_bbox(
                    (w, s, e, n), tags={'natural': 'water', 'waterway': ['river', 'stream']}
                )
                water_points = []
                for geom in water_gdf.geometry:
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
                        # coords are (lon, lat); subsample every 4th vertex
                        water_points.extend((y, x) for x, y in geom_coords[::4])
                    except Exception:
                        continue
            except Exception:
                water_points = []
            graph.graph['water_points'] = water_points

            # Fetch roadside viewpoints (tourism=viewpoint only — not natural=peak,
            # which are tagged on mountain tops far from roads).
            try:
                vp_gdf = osmnx.features_from_bbox(
                    (w, s, e, n), tags={'tourism': ['viewpoint', 'scenic_viewpoint']}
                )
                pts = vp_gdf.geometry.representative_point()
                viewpoints = list(zip(pts.y, pts.x))
            except Exception:
                viewpoints = []
            graph.graph['viewpoints'] = viewpoints

            # Fetch natural areas (forests, parks, meadows) for proximity scoring.
            try:
                nature_gdf = osmnx.features_from_bbox(
                    (w, s, e, n),
                    tags={
                        'natural': ['wood', 'scrub', 'heath', 'grassland'],
                        'landuse': ['forest', 'meadow', 'grass', 'village_green', 'recreation_ground'],
                        'leisure': ['park', 'nature_reserve'],
                    }
                )
                nature_points = []
                for geom in nature_gdf.geometry:
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
                        nature_points.extend((y, x) for x, y in geom_coords[::4])
                    except Exception:
                        continue
            except Exception:
                nature_points = []
            graph.graph['nature_points'] = nature_points

            os.makedirs(_CACHE_DIR, exist_ok=True)
            with open(cache_file, 'wb') as f:
                pickle.dump(graph, f)
        _GRAPH_CACHE[bbox_key] = graph

    graph = _GRAPH_CACHE[bbox_key]
    midpoint_lat = (n + s) / 2
    midpoint_lon = (e + w) / 2
    route_map = folium.Map(location=[midpoint_lat, midpoint_lon], zoom_start=10)
    return graph, route_map


def _curviness_score(data):
    geom = data.get('geometry')
    length = data.get('length', 0)
    if geom is None or length < 10:
        return 0.0
    coords = list(geom.coords)
    if len(coords) < 2:
        return 0.0
    dx = coords[-1][0] - coords[0][0]
    dy = coords[-1][1] - coords[0][1]
    lat_avg = (coords[0][1] + coords[-1][1]) / 2.0
    meters_per_lon = 111320 * math.cos(math.radians(lat_avg))
    straight_dist = math.sqrt((dx * meters_per_lon) ** 2 + (dy * 111320) ** 2)
    if straight_dist < 1:
        return 1.0
    return min(max(length / straight_dist - 1.0, 0.0), 1.0)


def _road_type_score(data):
    highway = data.get('highway', 'residential')
    if isinstance(highway, list):
        highway = highway[0]
    table = {
        'motorway': 0.15, 'trunk': 0.15, 'primary': 0.3,
        'motorway_link': 0.0, 'trunk_link': 0.0, 'primary_link': 0.1,
        'secondary': 0.4, 'secondary_link': 0.4,
        'tertiary': 0.8, 'tertiary_link': 0.8,
        'unclassified': 1.0, 'residential': 0.3,
        'living_street': 0.6, 'service': 0.1,
    }
    return table.get(highway, 0.4)



def _traffic_score(data):
    lanes = data.get('lanes')
    if lanes is None:
        return 0.4  # no lanes tag → quiet single-lane country road
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
    # Fixed component weights: speed, traffic, water, viewpoints.
    W_SPEED, W_TRAFFIC, W_WATER, W_VIEW = 1.0, 0.5, 1.0, 0.5

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
    view_abs = np.maximum(0.0, 1.0 - dists_to(view_tree) / 200.0)

    scores = np.empty(len(edges))
    for i, (u, v, k, data) in enumerate(edges):
        scores[i] = (
            w_curve * _curviness_score(data) +
            w_road * _road_type_score(data) +
            w_nature * float(nature_rel[i]) +
            W_SPEED * _speed_score(data) +
            W_TRAFFIC * _traffic_score(data) +
            W_WATER * float(water_rel[i]) +
            W_VIEW * float(view_abs[i])
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

    Adaptive detour search: probe each K in _K_LADDER (scenic_cost =
    travel_time * exp(K * (1 - scenic_t)); one Dijkstra each), keep candidates
    within the detour budget, and return the one with the best scenic gain per
    extra minute. The fast route is the floor candidate, so when no detour is
    worth it the fast route wins naturally (no cliff fallback).
    """
    fast_route = networkx.shortest_path(G, start, end, weight='travel_time')
    fast_time = networkx.path_weight(G, fast_route, weight='travel_time')
    fast_score = _tw_mean_score(G, fast_route)

    best_route, best_time, best_value = fast_route, fast_time, 0.0
    seen = {tuple(fast_route)}
    for K in _K_LADDER:
        for u, v, k, data in G.edges(data=True, keys=True):
            data['scenic_cost'] = data['travel_time'] * math.exp(K * (1.0 - data.get('scenic_t', 0.0)))
        try:
            route = networkx.shortest_path(G, start, end, weight='scenic_cost')
        except (networkx.NetworkXNoPath, networkx.NodeNotFound):
            break
        key = tuple(route)
        if key in seen:
            continue
        seen.add(key)
        time = networkx.path_weight(G, route, weight='travel_time')
        if time > fast_time * max_detour_factor:
            continue
        extra_min = max((time - fast_time) * 60, 1.0)
        value = (_tw_mean_score(G, route) - fast_score) / extra_min
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
