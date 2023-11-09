import pickle
import os
import osmnx

pickle_file = 'graph.pkl'

graph = osmnx.graph_from_xml("northeast.osm")
with open(pickle_file, 'wb') as f:
    pickle.dump(graph, f)