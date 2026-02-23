from flask import Flask, request, render_template
from geopy.distance import geodesic
from geopy.geocoders import Nominatim
import osmnx
import folium
import matplotlib
from route_planner import initialize_graph, score_scenic_edges, plan_scenic_route

app = Flask(__name__)
geolocator = Nominatim(user_agent="scenic_route_app")
DEFAULT_START = "312 Thurston Ave, Ithaca, NY 14850"
DEFAULT_END = "1781 Taughannock Blvd, Ulysses, NY 14886"

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        start_address = request.form.get('start_address') or DEFAULT_START
        end_address = request.form.get('end_address') or DEFAULT_END

        curviness_weight = float(request.form.get('curviness_weight', 1.0))
        nature_weight = float(request.form.get('nature_weight', 1.0))
        road_type_weight = float(request.form.get('road_type_weight', 1.0))
        max_detour = float(request.form.get('max_detour', 1.5))

        print(start_address)
        print(end_address)

        try:
            start_location = geolocator.geocode(start_address)
            end_location = geolocator.geocode(end_address)
        except:
            print("Error in geocoding")

        start_coords = (start_location.latitude, start_location.longitude)
        end_coords = (end_location.latitude, end_location.longitude)

        distance = geodesic(start_coords, end_coords).miles

        MAX_DISTANCE_MILES = 50
        if distance > MAX_DISTANCE_MILES:
            return render_template('index.html',
                error=f"Addresses are {distance:.0f} miles apart. Please keep routes under {MAX_DISTANCE_MILES} miles.",
                curviness_weight=curviness_weight,
                nature_weight=nature_weight,
                road_type_weight=road_type_weight,
                max_detour=max_detour,
            )

        graph, route_map = initialize_graph(start_coords, end_coords)

        start_node = osmnx.distance.nearest_nodes(graph, X=[start_coords[1]], Y=[start_coords[0]])
        end_node = osmnx.distance.nearest_nodes(graph, X=[end_coords[1]], Y=[end_coords[0]])

        weights = {
            'curviness': curviness_weight,
            'road_type': road_type_weight,
            'nature': nature_weight,
        }
        score_scenic_edges(graph, weights)

        fast_route, scenic_route = plan_scenic_route(
            graph, start_node[0], end_node[0], max_detour
        )

        fast_coords = [(graph.nodes[n]['y'], graph.nodes[n]['x']) for n in fast_route]
        scenic_coords = [(graph.nodes[n]['y'], graph.nodes[n]['x']) for n in scenic_route]

        folium.PolyLine(fast_coords, color="blue", weight=3, opacity=0.6, tooltip="Fastest").add_to(route_map)
        folium.PolyLine(scenic_coords, color="green", weight=4, opacity=0.9, tooltip="Scenic").add_to(route_map)

        start_marker = folium.Marker(location=[start_coords[0], start_coords[1]], popup=start_address)
        end_marker = folium.Marker(location=[end_coords[0], end_coords[1]], popup=end_address)
        start_marker.add_to(route_map)
        end_marker.add_to(route_map)
        map_html = route_map._repr_html_()

        return render_template(
            'index.html',
            distance=distance,
            route_map=map_html,
            curviness_weight=curviness_weight,
            nature_weight=nature_weight,
            road_type_weight=road_type_weight,
            max_detour=max_detour,
        )

    return render_template('index.html',
        curviness_weight=1.0,
        nature_weight=1.0,
        road_type_weight=1.0,
        max_detour=1.5,
    )

if __name__ == '__main__':
    app.run(debug=True, port=5001)
