import osmnx
import networkx
import folium
import math
import heapq
import copy
import os
import pickle
from collections import defaultdict

_GRAPH_CACHE: dict = {}
_CACHE_DIR = os.path.join(os.path.dirname(__file__), '.graph_cache')

# osmnx.config(
#     db_host='localhost',
#     db_name='map_data',
#     db_user='postgres',
#     db_password='oddopolis'
# )

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
        cache_file = os.path.join(_CACHE_DIR, f'graph_{bbox_key}.pkl')
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
                speed = None
                raw = data.get('maxspeed')
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
    dx = coords[-1][0] - coords[0][0]  # longitude diff
    dy = coords[-1][1] - coords[0][1]  # latitude diff
    lat_avg = (coords[0][1] + coords[-1][1]) / 2.0
    meters_per_lon = 111320 * math.cos(math.radians(lat_avg))
    straight_dist = math.sqrt((dx * meters_per_lon) ** 2 + (dy * 111320) ** 2)
    if straight_dist < 1:
        return 1.0
    sinuosity = length / straight_dist
    return min(max(sinuosity - 1.0, 0.0), 1.0)


def _road_type_score(data):
    highway = data.get('highway', 'residential')
    if isinstance(highway, list):
        highway = highway[0]
    table = {
        'motorway': 0.0, 'trunk': 0.0, 'primary': 0.3,
        'motorway_link': 0.0, 'trunk_link': 0.0, 'primary_link': 0.1,
        'secondary': 0.4, 'secondary_link': 0.4,
        'tertiary': 0.8, 'tertiary_link': 0.8,
        'unclassified': 1.0, 'residential': 0.3,
        'living_street': 0.6, 'service': 0.1,
    }
    return table.get(highway, 0.4)


def _nature_score(data):
    score = 0.0
    if data.get('route') == 'scenic':
        score += 0.6
    if data.get('tourism') in ('viewpoint', 'attraction'):
        score += 0.4
    if data.get('natural') in ('wood', 'water', 'coastline'):
        score += 0.3
    if data.get('landuse') in ('forest', 'meadow'):
        score += 0.2
    return min(score, 1.0)


def _traffic_score(data):
    lanes = data.get('lanes')
    if lanes is None:
        # No lanes tag: almost certainly a quiet single-lane country road
        return 0.4
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
    # Peaks at 45-55 mph (open country road). Penalizes slow urban roads and
    # very fast highways where you're just a number in traffic.
    if speed >= 65:
        return 0.4
    elif speed >= 55:
        return 1.0
    elif speed >= 45:
        return 0.8
    elif speed >= 35:
        return 0.5
    else:
        return 0.2  # slow rural roads are fine; only urban arterials score 0


def score_scenic_edges(G, weights: dict) -> None:
    w_curve = weights.get('curviness', 1.0)
    w_road = weights.get('road_type', 1.0)
    w_nature = weights.get('nature', 1.0)
    # Speed and traffic are always included at weight 1.0.
    max_score = w_curve + w_road + w_nature + 2.0

    for u, v, k, data in G.edges(data=True, keys=True):
        scenic_score = (
            w_curve * _curviness_score(data) +
            w_road * _road_type_score(data) +
            w_nature * _nature_score(data) +
            _speed_score(data) +
            _traffic_score(data)
        )
        data['scenic_score'] = scenic_score
        # Exponential penalty: unscenic roads cost much more, scenic roads much less.
        # Range: exp(2)≈7.4× for score=0 (highway) down to 1× for max scenic.
        t = scenic_score / max_score if max_score > 0 else 0.0
        data['scenic_cost'] = data['travel_time'] * math.exp(2.0 * (1.0 - t))


def plan_route(G, start, end):
    return networkx.shortest_path(G, start, end, weight="travel_time")


def plan_scenic_route(G, start, end, max_detour_factor) -> tuple:
    fast_route = networkx.shortest_path(G, start, end, weight='travel_time')
    fast_time = networkx.path_weight(G, fast_route, weight='travel_time')

    try:
        scenic_route = networkx.shortest_path(G, start, end, weight='scenic_cost')
    except (networkx.NetworkXNoPath, networkx.NodeNotFound):
        return fast_route, fast_route

    scenic_time = networkx.path_weight(G, scenic_route, weight='travel_time')
    if scenic_time > fast_time * max_detour_factor:
        return fast_route, fast_route

    return fast_route, scenic_route


def yen_k_shortest_routes(G, start, end, weight, k):
    shortest_paths = []
    shortest_paths.append(networkx.shortest_path(G, start, end, weight))

    for k in range(1, k):
        for i in range(len(shortest_paths[-1]) - 1):
            path_list = []
            current_path = shortest_paths[-1]
            spur_node = current_path[i]
            root_path = current_path[:i + 1]

            graph_copy = copy.deepcopy(G)
            graph_copy.remove_nodes_from(root_path[:-1])

            try:
                spur_path = networkx.shortest_path(graph_copy, spur_node, end, weight)
            except networkx.NetworkXNoPath:
                continue

            full_path = root_path + spur_path[1:]

            if full_path not in shortest_paths:
                path_list.append((full_path, networkx.path_weight(G, full_path, weight)))

        if path_list:
            path_list.sort(key=lambda e: e[1])
            shortest_paths.append(path_list[0][0])

    return shortest_paths
