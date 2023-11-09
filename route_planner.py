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

def initialize_graph(start, end):
  buffer = 0.1
  n = max(start[0], end[0]) + buffer
  s = min(start[0], end[0]) - buffer
  e = max(start[1], end[1]) + buffer
  w = min(start[1], end[1]) - buffer
  box = [n, s, e, w]
  #print(f"Start: {start}")
  #print(f"End: {end}")
  #print(f"Bounding Box: {box}")
  private_filter = '["area"!~"yes"]["highway"~"motorway|trunk|primary|secondary|tertiary|unclassified|residential|service|motorway_link|trunk_link|primary_link|secondary_link|tertiary_link"]["access"!~"private"]'
  graph = osmnx.graph_from_bbox(*box, simplify=False, network_type='drive', custom_filter=private_filter)
  osmnx.plot_graph(graph)
  midpoint_lat = (n+s)/2
  midpoint_lon = (e+w)/2
  route_map = folium.Map(location=[midpoint_lat,midpoint_lon], zoom_start=10)
  
  for u, v, k, data in graph.edges(data=True, keys=True):
    if type(data.get('maxspeed')) == str:
      speed_data = data.get('maxspeed').split()
      data.update({'maxspeed': float(speed_data[0])})
    data['travel_time'] = data['length'] / 1609.34 / data.get('maxspeed')
  return graph, route_map

def scenic_factorify(G):
  scenic_factor = 0
  return scenic_factor

def plan_route(G, start, end):
  route = networkx.shortest_path(G, start, end, weight="travel_time")
  osmnx.plot_graph_route(G, route)
  return route
















def yen_k_shortest_routes(G, start, end, weight, k):
  shortest_paths = []
  shortest_paths.append(networkx.shortest_path(G, start, end, weight))

  for k in range(1, k):
    for i in range(len(shortest_paths[-1]) - 1):
      path_list = []
      current_path = shortest_paths[-1]
      spur_node = current_path[i]
      root_path = current_path[:i+1]

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
  
#def routing_algorithm(G, start, end, weight):


  



