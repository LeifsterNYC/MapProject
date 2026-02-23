import osmnx
import networkx
import folium
import math
import heapq
import copy
from collections import defaultdict

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
    box = [n, s, e, w]
    private_filter = '["area"!~"yes"]["highway"~"motorway|trunk|primary|secondary|tertiary|unclassified|residential|service|motorway_link|trunk_link|primary_link|secondary_link|tertiary_link"]["access"!~"private"]'
    graph = osmnx.graph_from_bbox(*box, simplify=False, network_type='drive', custom_filter=private_filter)
    osmnx.plot_graph(graph)
    midpoint_lat = (n + s) / 2
    midpoint_lon = (e + w) / 2
    route_map = folium.Map(location=[midpoint_lat, midpoint_lon], zoom_start=10)

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
    straight_dist = math.sqrt(dx * dx + dy * dy) * 111320
    if straight_dist < 1:
        return 1.0
    sinuosity = length / straight_dist
    return min(sinuosity - 1.0, 1.0)


def _road_type_score(data):
    highway = data.get('highway', 'residential')
    if isinstance(highway, list):
        highway = highway[0]
    table = {
        'motorway': 0.0, 'trunk': 0.05, 'primary': 0.2,
        'motorway_link': 0.0, 'trunk_link': 0.05, 'primary_link': 0.2,
        'secondary': 0.5, 'secondary_link': 0.5,
        'tertiary': 0.75, 'tertiary_link': 0.75,
        'unclassified': 0.9, 'residential': 0.7,
        'living_street': 0.8, 'service': 0.4,
    }
    return table.get(highway, 0.5)


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


def score_scenic_edges(G, weights: dict) -> None:
    w_curve = weights.get('curviness', 1.0)
    w_road = weights.get('road_type', 1.0)
    w_nature = weights.get('nature', 1.0)
    total_w = w_curve + w_road + w_nature or 1.0

    for u, v, k, data in G.edges(data=True, keys=True):
        scenic_score = (
            w_curve * _curviness_score(data) +
            w_road * _road_type_score(data) +
            w_nature * _nature_score(data)
        ) / total_w
        data['scenic_score'] = scenic_score
        data['scenic_cost'] = data['travel_time'] / (1.0 + scenic_score)


def plan_route(G, start, end):
    route = networkx.shortest_path(G, start, end, weight="travel_time")
    osmnx.plot_graph_route(G, route)
    return route


def plan_scenic_route(G, start, end, max_detour_factor) -> tuple:
    fast_route = networkx.shortest_path(G, start, end, weight='travel_time')
    fast_time = networkx.path_weight(G, fast_route, weight='travel_time')

    def heuristic(u, v):
        dlat = G.nodes[u]['y'] - G.nodes[v]['y']
        dlon = G.nodes[u]['x'] - G.nodes[v]['x']
        return math.sqrt(dlat ** 2 + dlon ** 2) * 69.0 / 65.0

    try:
        scenic_route = networkx.astar_path(G, start, end, heuristic=heuristic, weight='scenic_cost')
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
