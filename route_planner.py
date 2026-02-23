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
    private_filter = '["area"!~"yes"]["highway"~"motorway|trunk|primary|secondary|tertiary|unclassified|residential|service|motorway_link|trunk_link|primary_link|secondary_link|tertiary_link"]["access"!~"private"]'
    graph = osmnx.graph_from_bbox((w, s, e, n), simplify=True, network_type='drive', custom_filter=private_filter)
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
        'motorway': 0.0, 'trunk': 0.0, 'primary': 0.05,
        'motorway_link': 0.0, 'trunk_link': 0.0, 'primary_link': 0.05,
        'secondary': 0.5, 'secondary_link': 0.5,
        'tertiary': 0.8, 'tertiary_link': 0.8,
        'unclassified': 1.0, 'residential': 0.8,
        'living_street': 0.9, 'service': 0.3,
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

    for u, v, k, data in G.edges(data=True, keys=True):
        # scores are summed (not normalized) so weights stack —
        # default (1,1,1) gives scenic_score up to ~3, enough to overcome detour cost
        scenic_score = (
            w_curve * _curviness_score(data) +
            w_road * _road_type_score(data) +
            w_nature * _nature_score(data)
        )
        data['scenic_score'] = scenic_score
        data['scenic_cost'] = data['travel_time'] / (1.0 + scenic_score)


def plan_route(G, start, end):
    return networkx.shortest_path(G, start, end, weight="travel_time")


def plan_scenic_route(G, start, end, max_detour_factor) -> tuple:
    fast_route = networkx.shortest_path(G, start, end, weight='travel_time')
    fast_time = networkx.path_weight(G, fast_route, weight='travel_time')

    def scenic_miles(route):
        total = 0.0
        for u, v in zip(route, route[1:]):
            d = list(G[u][v].values())[0]
            total += d.get('scenic_score', 0) * d.get('length', 0)
        return total

    candidates = []

    # Dijkstra on scenic_cost — globally optimal cost path
    try:
        dijk = networkx.shortest_path(G, start, end, weight='scenic_cost')
        if networkx.path_weight(G, dijk, 'travel_time') <= fast_time * max_detour_factor:
            candidates.append(dijk)
    except (networkx.NetworkXNoPath, networkx.NodeNotFound):
        pass

    # Also try routing through the top scenic edge endpoints as waypoints.
    # Dijkstra alone misses roads that require a detour to reach even when
    # they have high scenic value (e.g. a curvy backroad off the main corridor).
    top_edges = sorted(
        G.edges(data=True, keys=True),
        key=lambda e: e[3].get('scenic_score', 0) * e[3].get('length', 0),
        reverse=True,
    )[:10]

    seen = set()
    for u, v, k, data in top_edges:
        for wp in (u, v):
            if wp in (start, end) or wp in seen:
                continue
            seen.add(wp)
            try:
                leg1 = networkx.shortest_path(G, start, wp, weight='travel_time')
                leg2 = networkx.shortest_path(G, wp, end, weight='travel_time')
                route = leg1 + leg2[1:]
                if networkx.path_weight(G, route, 'travel_time') <= fast_time * max_detour_factor:
                    candidates.append(route)
            except (networkx.NetworkXNoPath, networkx.NodeNotFound):
                continue

    if not candidates:
        return fast_route, fast_route

    return fast_route, max(candidates, key=scenic_miles)


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
