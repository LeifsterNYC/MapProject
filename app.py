from flask import Flask, request, render_template, Response, stream_with_context
from geopy.distance import geodesic
from geopy.geocoders import Nominatim
import osmnx
import folium
import matplotlib
import json
from route_planner import initialize_graph, score_scenic_edges, plan_scenic_route

app = Flask(__name__)
geolocator = Nominatim(user_agent="scenic_route_app")
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


def _compute_route(start_address, end_address, curviness_weight, nature_weight, road_type_weight, max_detour, emit):
    """Run the full pipeline, calling emit(stage_str) at each step. Returns render_template result."""
    emit('Geocoding addresses...')
    start_location = geolocator.geocode(start_address)
    end_location = geolocator.geocode(end_address)
    start_coords = (start_location.latitude, start_location.longitude)
    end_coords = (end_location.latitude, end_location.longitude)
    distance = geodesic(start_coords, end_coords).miles

    if distance > MAX_DISTANCE_MILES:
        raise ValueError(f"Addresses are {distance:.0f} miles apart. Please keep routes under {MAX_DISTANCE_MILES} miles.")

    emit('Fetching road network...')
    graph, route_map = initialize_graph(start_coords, end_coords)
    start_node = osmnx.distance.nearest_nodes(graph, X=[start_coords[1]], Y=[start_coords[0]])[0]
    end_node = osmnx.distance.nearest_nodes(graph, X=[end_coords[1]], Y=[end_coords[0]])[0]

    emit('Scoring scenic edges...')
    score_scenic_edges(graph, {'curviness': curviness_weight, 'road_type': road_type_weight, 'nature': nature_weight})

    emit('Computing routes...')
    fast_route, scenic_route = plan_scenic_route(graph, start_node, end_node, max_detour)

    fast_coords = [(graph.nodes[n]['y'], graph.nodes[n]['x']) for n in fast_route]
    scenic_coords = [(graph.nodes[n]['y'], graph.nodes[n]['x']) for n in scenic_route]
    folium.PolyLine(fast_coords, color="blue", weight=3, opacity=0.6, tooltip="Fastest").add_to(route_map)
    folium.PolyLine(scenic_coords, color="green", weight=4, opacity=0.9, tooltip="Scenic").add_to(route_map)
    folium.Marker(location=list(start_coords), popup=start_address).add_to(route_map)
    folium.Marker(location=list(end_coords), popup=end_address).add_to(route_map)
    return distance, route_map._repr_html_(), _gmaps_url(fast_coords), _gmaps_url(scenic_coords)


@app.route('/stream')
def stream():
    start_address = request.args.get('start_address') or DEFAULT_START
    end_address = request.args.get('end_address') or DEFAULT_END
    curviness_weight = float(request.args.get('curviness_weight', 1.0))
    nature_weight = float(request.args.get('nature_weight', 1.0))
    road_type_weight = float(request.args.get('road_type_weight', 1.0))
    max_detour = float(request.args.get('max_detour', 3.0))

    def generate():
        def event(data):
            return 'data: ' + json.dumps(data) + '\n\n'

        try:
            stages = []

            def emit(stage):
                stages.append(stage)

            # We can't yield from inside _compute_route, so we emit stages after the fact.
            # Instead, inline the pipeline here so we can yield between real steps.
            yield event({'stage': 'Geocoding addresses...'})
            start_location = geolocator.geocode(start_address)
            end_location = geolocator.geocode(end_address)
            start_coords = (start_location.latitude, start_location.longitude)
            end_coords = (end_location.latitude, end_location.longitude)
            distance = geodesic(start_coords, end_coords).miles

            if distance > MAX_DISTANCE_MILES:
                yield event({'error': f"Addresses are {distance:.0f} miles apart. Please keep routes under {MAX_DISTANCE_MILES} miles."})
                return

            yield event({'stage': 'Fetching road network...'})
            graph, route_map = initialize_graph(start_coords, end_coords)
            start_node = osmnx.distance.nearest_nodes(graph, X=[start_coords[1]], Y=[start_coords[0]])[0]
            end_node = osmnx.distance.nearest_nodes(graph, X=[end_coords[1]], Y=[end_coords[0]])[0]

            yield event({'stage': 'Scoring scenic edges...'})
            score_scenic_edges(graph, {'curviness': curviness_weight, 'road_type': road_type_weight, 'nature': nature_weight})

            yield event({'stage': 'Computing routes...'})
            fast_route, scenic_route = plan_scenic_route(graph, start_node, end_node, max_detour)

            fast_coords = [(graph.nodes[n]['y'], graph.nodes[n]['x']) for n in fast_route]
            scenic_coords = [(graph.nodes[n]['y'], graph.nodes[n]['x']) for n in scenic_route]
            folium.PolyLine(fast_coords, color="blue", weight=3, opacity=0.6, tooltip="Fastest").add_to(route_map)
            folium.PolyLine(scenic_coords, color="green", weight=4, opacity=0.9, tooltip="Scenic").add_to(route_map)
            folium.Marker(location=list(start_coords), popup=start_address).add_to(route_map)
            folium.Marker(location=list(end_coords), popup=end_address).add_to(route_map)
            map_html = route_map._repr_html_()

            html = render_template('index.html',
                distance=distance, route_map=map_html,
                fast_gmaps=_gmaps_url(fast_coords),
                scenic_gmaps=_gmaps_url(scenic_coords),
                curviness_weight=curviness_weight, nature_weight=nature_weight,
                road_type_weight=road_type_weight, max_detour=max_detour)

            yield event({'done': True, 'html': html})

        except Exception as e:
            yield event({'error': str(e)})

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

        try:
            distance, map_html, fast_gmaps, scenic_gmaps = _compute_route(
                start_address, end_address,
                curviness_weight, nature_weight, road_type_weight, max_detour,
                emit=lambda s: None,
            )
        except ValueError as e:
            return render_template('index.html', error=str(e),
                curviness_weight=curviness_weight, nature_weight=nature_weight,
                road_type_weight=road_type_weight, max_detour=max_detour)

        return render_template('index.html',
            distance=distance, route_map=map_html,
            fast_gmaps=fast_gmaps, scenic_gmaps=scenic_gmaps,
            curviness_weight=curviness_weight, nature_weight=nature_weight,
            road_type_weight=road_type_weight, max_detour=max_detour)

    return render_template('index.html',
        curviness_weight=1.0, nature_weight=1.0,
        road_type_weight=1.0, max_detour=3.0)


if __name__ == '__main__':
    app.run(debug=True, port=5001, use_reloader=False, threaded=False)
