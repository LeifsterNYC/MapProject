from flask import Flask, request, render_template
from geopy.distance import geodesic
from geopy.geocoders import Nominatim
import osmnx
import folium
import matplotlib
from route_planner import initialize_graph, scenic_factorify, plan_route

app = Flask(__name__)
geolocator = Nominatim(user_agent="scenic_route_app")
DEFAULT_START = "312 Thurston Ave, Ithaca, NY 14850"
DEFAULT_END = "1781 Taughannock Blvd, Ulysses, NY 14886"

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        start_address = request.form.get('start_address') or DEFAULT_START
        end_address = request.form.get('end_address') or DEFAULT_END
        
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

        graph, route_map = initialize_graph(start_coords, end_coords)

        start_node = osmnx.distance.nearest_nodes(graph, X=[start_coords[1]], Y=[start_coords[0]])
        end_node = osmnx.distance.nearest_nodes(graph, X=[end_coords[1]], Y=[end_coords[0]])

        route = plan_route(graph, start_node[0], end_node[0])

        route_coords = [(graph.nodes[node]['y'], graph.nodes[node]['x']) for node in route]
        folium.PolyLine(route_coords, color="blue", weight=2.5, opacity=1).add_to(route_map)

        start_marker = folium.Marker(location=[start_coords[0], start_coords[1]], popup=start_address)
        end_marker = folium.Marker(location=[end_coords[0], end_coords[1]], end=start_address)
        start_marker.add_to(route_map)
        end_marker.add_to(route_map)
        map_html = route_map._repr_html_()

        return render_template('index.html', distance=distance, route_map=map_html)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)