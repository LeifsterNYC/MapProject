from dotenv import load_dotenv
load_dotenv()
from flask import Flask, request, render_template, Response, stream_with_context
from geopy.distance import geodesic
from geopy.geocoders import Nominatim
import osmnx
import folium
import json
import queue
import threading
import os
import math
import requests
import numpy as np
from scipy.spatial import KDTree
from route_planner import initialize_graph, score_scenic_edges, plan_scenic_route

app = Flask(__name__)
geolocator = Nominatim(user_agent="ramble_app")
DEFAULT_START = "312 Thurston Ave, Ithaca, NY 14850"
DEFAULT_END = "1781 Taughannock Blvd, Ulysses, NY 14886"
MAX_DISTANCE_MILES = 50


def _gmaps_url(coords, max_waypoints=8):
    """Build a Google Maps directions URL from a list of (lat, lon) coords."""
    if len(coords) < 2:
        return None
    origin = f"{coords[0][0]:.6f},{coords[0][1]:.6f}"
    destination = f"{coords[-1][0]:.6f},{coords[-1][1]:.6f}"
    intermediate = coords[1:-1]
    if len(intermediate) > max_waypoints:
        # Evenly sample to stay within the waypoint limit
        step = (len(intermediate) - 1) / (max_waypoints - 1)
        intermediate = [intermediate[round(i * step)] for i in range(max_waypoints)]
    waypoints = '|'.join(f"{lat:.6f},{lon:.6f}" for lat, lon in intermediate)
    url = (f"https://www.google.com/maps/dir/?api=1"
           f"&origin={origin}&destination={destination}&travelmode=driving")
    if waypoints:
        url += f"&waypoints={waypoints}"
    return url


def _route_coords(G, route):
    """Extract (lat, lon) coords following actual road geometry, not just node points."""
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
        # Skip the first point on all but the first edge to avoid duplicates
        coords.extend(edge_coords if not coords else edge_coords[1:])
    if not coords and route:
        coords.append((G.nodes[route[0]]['y'], G.nodes[route[0]]['x']))
    return coords


_TOMTOM_KEY = os.environ.get('TOMTOM_API_KEY')


def _fetch_closure_points(w, s, e, n):
    """Return [(lat, lon), ...] midpoints of road-closure incidents in the bbox."""
    if not _TOMTOM_KEY:
        return []
    # Approximate km² check: 1° lat ≈ 111 km, 1° lon ≈ 111·cos(lat) km
    mid_lat = (n + s) / 2
    area_km2 = (e - w) * 111 * math.cos(math.radians(mid_lat)) * (n - s) * 111
    if area_km2 > 9500:
        return []
    url = 'https://api.tomtom.com/traffic/services/5/incidentDetails'
    params = {
        'key': _TOMTOM_KEY,
        'bbox': f'{w},{s},{e},{n}',
        'fields': '{incidents{geometry{type,coordinates},properties{iconCategory}}}',
    }
    try:
        resp = requests.get(url, params=params, timeout=5)
        resp.raise_for_status()
        incidents = resp.json().get('incidents', [])
    except Exception:
        return []
    points = []
    for feat in incidents:
        if feat.get('properties', {}).get('iconCategory') != 8:
            continue
        geom = feat.get('geometry', {})
        coords = geom.get('coordinates', [])
        if not coords:
            continue
        if geom.get('type') == 'LineString':
            mid = coords[len(coords) // 2]
        else:
            mid = coords  # Point
        points.append((mid[1], mid[0]))  # (lat, lon)
    return points


def _apply_closures(G, closure_points):
    """Set travel_time=inf on edges near closure points. Returns dict of original values."""
    if not closure_points:
        return {}
    node_data = list(G.nodes(data=True))[:200]
    ref_lat = sum(d.get('y', 0) for _, d in node_data) / max(len(node_data), 1)
    lat_m = 111320.0
    lon_m = 111320.0 * math.cos(math.radians(ref_lat))
    arr = np.array([(lat * lat_m, lon * lon_m) for lat, lon in closure_points])
    tree = KDTree(arr)
    saved = {}
    for u, v, k, data in G.edges(data=True, keys=True):
        geom = data.get('geometry')
        if geom is not None:
            coords = list(geom.coords)
            mid = coords[len(coords) // 2]
            mid_lon, mid_lat = mid[0], mid[1]
        else:
            mid_lat = (G.nodes[u]['y'] + G.nodes[v]['y']) / 2
            mid_lon = (G.nodes[u]['x'] + G.nodes[v]['x']) / 2
        dist, _ = tree.query([mid_lat * lat_m, mid_lon * lon_m])
        if dist < 75:  # 75 m match radius
            saved[(u, v, k)] = data.get('travel_time')
            data['travel_time'] = float('inf')
    return saved


def _restore_closures(G, saved):
    """Restore travel_time values after routing (graph is cached between requests)."""
    for (u, v, k), tt in saved.items():
        G[u][v][k]['travel_time'] = tt


def _pipeline(start_address, end_address, curviness_weight, nature_weight, road_type_weight, max_detour, on_stage=None, use_tomtom=False):
    """Run the full route pipeline. Calls on_stage(str) at each step if provided.
    Returns (distance, map_html, fast_gmaps, scenic_gmaps, fast_min, scenic_min, closure_count)."""
    if on_stage:
        on_stage('Geocoding addresses...')
    start_location = geolocator.geocode(start_address)
    if start_location is None:
        raise ValueError(f"Could not find address: {start_address}")
    end_location = geolocator.geocode(end_address)
    if end_location is None:
        raise ValueError(f"Could not find address: {end_address}")
    start_coords = (start_location.latitude, start_location.longitude)
    end_coords = (end_location.latitude, end_location.longitude)
    distance = geodesic(start_coords, end_coords).miles

    if distance > MAX_DISTANCE_MILES:
        raise ValueError(f"Addresses are {distance:.0f} miles apart. Please keep routes under {MAX_DISTANCE_MILES} miles.")

    if on_stage:
        on_stage('Fetching road network...')
    graph, route_map = initialize_graph(start_coords, end_coords)
    start_node = osmnx.distance.nearest_nodes(graph, X=[start_coords[1]], Y=[start_coords[0]])[0]
    end_node = osmnx.distance.nearest_nodes(graph, X=[end_coords[1]], Y=[end_coords[0]])[0]

    # Apply real-time road closures before scoring/routing so both routes avoid them
    buffer = 0.1
    n = max(start_coords[0], end_coords[0]) + buffer
    s = min(start_coords[0], end_coords[0]) - buffer
    e = max(start_coords[1], end_coords[1]) + buffer
    w = min(start_coords[1], end_coords[1]) - buffer
    saved_times = {}
    if use_tomtom:
        if on_stage:
            on_stage('Fetching road closures...')
        closure_pts = _fetch_closure_points(w, s, e, n)
        saved_times = _apply_closures(graph, closure_pts)

    if on_stage:
        on_stage('Scoring scenic edges...')
    score_scenic_edges(graph, {'curviness': curviness_weight, 'road_type': road_type_weight, 'nature': nature_weight})

    if on_stage:
        on_stage('Computing routes...')
    fast_route, scenic_route, fast_min, scenic_min = plan_scenic_route(graph, start_node, end_node, max_detour)
    fast_coords = _route_coords(graph, fast_route)
    scenic_coords = _route_coords(graph, scenic_route)
    folium.PolyLine(fast_coords, color="blue", weight=3, opacity=0.6, tooltip="Fastest").add_to(route_map)
    folium.PolyLine(scenic_coords, color="green", weight=4, opacity=0.9, tooltip="Scenic").add_to(route_map)
    folium.Marker(location=list(start_coords), popup=start_address).add_to(route_map)
    folium.Marker(location=list(end_coords), popup=end_address).add_to(route_map)

    if saved_times:
        _restore_closures(graph, saved_times)

    return distance, route_map._repr_html_(), _gmaps_url(fast_coords), _gmaps_url(scenic_coords), fast_min, scenic_min, len(saved_times)


@app.route('/stream')
def stream():
    start_address = request.args.get('start_address') or DEFAULT_START
    end_address = request.args.get('end_address') or DEFAULT_END
    curviness_weight = float(request.args.get('curviness_weight', 1.0))
    nature_weight = float(request.args.get('nature_weight', 1.0))
    road_type_weight = float(request.args.get('road_type_weight', 1.0))
    max_detour = float(request.args.get('max_detour', 3.0))
    use_tomtom = bool(request.args.get('use_tomtom')) and bool(_TOMTOM_KEY)

    q = queue.SimpleQueue()

    def run():
        try:
            result = _pipeline(
                start_address, end_address,
                curviness_weight, nature_weight, road_type_weight, max_detour,
                on_stage=lambda msg: q.put({'stage': msg}),
                use_tomtom=use_tomtom,
            )
            q.put({'result': result})
        except Exception as e:
            q.put({'error': str(e)})

    threading.Thread(target=run, daemon=True).start()

    def generate():
        def event(data):
            return 'data: ' + json.dumps(data) + '\n\n'

        while True:
            item = q.get()
            if 'stage' in item:
                yield event({'stage': item['stage']})
            elif 'error' in item:
                yield event({'error': item['error']})
                return
            else:
                distance, map_html, fast_gmaps, scenic_gmaps, fast_min, scenic_min, closure_count = item['result']
                html = render_template('index.html',
                    distance=distance, route_map=map_html,
                    fast_gmaps=fast_gmaps, scenic_gmaps=scenic_gmaps,
                    fast_min=fast_min, scenic_min=scenic_min,
                    curviness_weight=curviness_weight, nature_weight=nature_weight,
                    road_type_weight=road_type_weight, max_detour=max_detour,
                    has_tomtom=bool(_TOMTOM_KEY), use_tomtom=use_tomtom,
                    closure_count=closure_count)
                yield event({'done': True, 'html': html})
                return

    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'},
    )


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        start_address = request.form.get('start_address') or DEFAULT_START
        end_address = request.form.get('end_address') or DEFAULT_END
        curviness_weight = float(request.form.get('curviness_weight', 1.0))
        nature_weight = float(request.form.get('nature_weight', 1.0))
        road_type_weight = float(request.form.get('road_type_weight', 1.0))
        max_detour = float(request.form.get('max_detour', 3.0))
        use_tomtom = bool(request.form.get('use_tomtom')) and bool(_TOMTOM_KEY)

        try:
            distance, map_html, fast_gmaps, scenic_gmaps, fast_min, scenic_min, closure_count = _pipeline(
                start_address, end_address,
                curviness_weight, nature_weight, road_type_weight, max_detour,
                use_tomtom=use_tomtom,
            )
        except ValueError as e:
            return render_template('index.html', error=str(e),
                curviness_weight=curviness_weight, nature_weight=nature_weight,
                road_type_weight=road_type_weight, max_detour=max_detour,
                has_tomtom=bool(_TOMTOM_KEY), use_tomtom=use_tomtom)

        return render_template('index.html',
            distance=distance, route_map=map_html,
            fast_gmaps=fast_gmaps, scenic_gmaps=scenic_gmaps,
            fast_min=fast_min, scenic_min=scenic_min,
            curviness_weight=curviness_weight, nature_weight=nature_weight,
            road_type_weight=road_type_weight, max_detour=max_detour,
            has_tomtom=bool(_TOMTOM_KEY), use_tomtom=use_tomtom,
            closure_count=closure_count)

    return render_template('index.html',
        curviness_weight=1.0, nature_weight=1.0,
        road_type_weight=1.0, max_detour=3.0,
        has_tomtom=bool(_TOMTOM_KEY), use_tomtom=False)


if __name__ == '__main__':
    app.run(debug=True, port=5001, use_reloader=False, threaded=False)
