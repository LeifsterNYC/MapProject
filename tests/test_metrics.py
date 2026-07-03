import networkx as nx

from evaluation.metrics import (
    geo_dist_m, point_to_polyline_m, waypoint_coverage,
    shared_node_fraction, shared_length_fraction,
    detour_ratio, mean_scenic_score,
)


def test_geo_dist_one_degree_lat_is_about_111km():
    d = geo_dist_m((0.0, 0.0), (1.0, 0.0))
    assert 110_000 < d < 112_000


def test_geo_dist_zero_for_same_point():
    assert geo_dist_m((42.44, -76.5), (42.44, -76.5)) == 0.0


def test_point_to_polyline_picks_nearest_vertex():
    point = (0.0, 0.0)
    polyline = [(0.0, 1.0), (0.0, 0.5), (0.0, 2.0)]  # nearest is (0, 0.5)
    d = point_to_polyline_m(point, polyline)
    assert 55_000 < d < 56_000


def test_waypoint_coverage_all_hit():
    via = [((0.0, 0.0), 300.0), ((0.0, 1.0), 300.0)]
    polyline = [(0.0, 0.0), (0.0, 0.5), (0.0, 1.0)]
    assert waypoint_coverage(via, polyline) == 1.0


def test_waypoint_coverage_half_hit():
    via = [((0.0, 0.0), 300.0), ((10.0, 10.0), 300.0)]  # second is ~1500 km away
    polyline = [(0.0, 0.0), (0.0, 0.001)]
    assert waypoint_coverage(via, polyline) == 0.5


def test_waypoint_coverage_respects_per_point_threshold():
    # ~111 m from the polyline: misses at 50 m threshold, hits at 300 m.
    via_tight = [((0.001, 0.0), 50.0)]
    via_loose = [((0.001, 0.0), 300.0)]
    polyline = [(0.0, -0.001), (0.0, 0.001)]
    assert waypoint_coverage(via_tight, polyline) == 0.0
    assert waypoint_coverage(via_loose, polyline) == 1.0


def test_waypoint_coverage_empty_via_is_none():
    assert waypoint_coverage([], [(0.0, 0.0)]) is None


def test_shared_node_fraction_identical():
    assert shared_node_fraction([1, 2, 3], [1, 2, 3]) == 1.0


def test_shared_node_fraction_disjoint():
    assert shared_node_fraction([1, 2, 3], [4, 5, 6]) == 0.0


def test_shared_node_fraction_partial():
    # union {1,2,3,4}, shared {2,3} -> 0.5
    assert shared_node_fraction([1, 2, 3], [2, 3, 4]) == 0.5


def test_shared_node_fraction_empty_scenic():
    assert shared_node_fraction([1, 2, 3], []) == 0.0


def test_detour_ratio():
    assert detour_ratio(10.0, 15.0) == 1.5


def test_detour_ratio_zero_fast_is_one():
    assert detour_ratio(0.0, 5.0) == 1.0


def _tiny_graph():
    # Path 1-2-3-4 with a parallel bypass edge 2-4.
    G = nx.MultiDiGraph()
    G.add_edge(1, 2, length=100.0, scenic_score=0.2, travel_time=1.0)
    G.add_edge(2, 3, length=100.0, scenic_score=0.8, travel_time=1.0)
    G.add_edge(3, 4, length=100.0, scenic_score=0.8, travel_time=1.0)
    G.add_edge(2, 4, length=300.0, scenic_score=0.1, travel_time=1.0)
    return G


def test_shared_length_fraction_partial_overlap():
    G = _tiny_graph()
    fast = [1, 2, 4]
    scenic = [1, 2, 3, 4]
    # shared length along scenic = 100 (edge 1-2); total scenic length = 300
    assert shared_length_fraction(G, fast, scenic) == 100.0 / 300.0


def test_mean_scenic_score():
    G = _tiny_graph()
    scenic = [1, 2, 3, 4]  # scores 0.2, 0.8, 0.8, uniform travel_time
    assert abs(mean_scenic_score(G, scenic) - 0.6) < 1e-9


def test_mean_scenic_score_is_time_weighted():
    G = nx.MultiDiGraph()
    G.add_edge(1, 2, length=100.0, scenic_score=1.0, travel_time=3.0)
    G.add_edge(2, 3, length=100.0, scenic_score=0.0, travel_time=1.0)
    # (1.0*3 + 0.0*1) / 4 = 0.75
    assert abs(mean_scenic_score(G, [1, 2, 3]) - 0.75) < 1e-9
