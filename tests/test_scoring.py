import numpy as np

from route_planner import relative_proximity_scores


def test_relative_proximity_ranks_closest_highest():
    scores = relative_proximity_scores([10.0, 20.0, 30.0])
    assert scores[0] == 1.0   # closest
    assert scores[2] == 0.0   # farthest
    assert 0.4 < scores[1] < 0.6


def test_relative_proximity_all_far_gates_to_zero():
    # median distance beyond the far gate -> feature is irrelevant here
    scores = relative_proximity_scores([5000.0, 6000.0, 7000.0], far_gate_m=2000.0)
    assert np.all(scores == 0.0)


def test_relative_proximity_all_infinite_gates_to_zero():
    scores = relative_proximity_scores([np.inf, np.inf])
    assert np.all(scores == 0.0)


def test_relative_proximity_empty_returns_empty():
    scores = relative_proximity_scores([])
    assert len(scores) == 0


import networkx as nx
from route_planner import plan_scenic_route, _road_type_score


def _detour_graph():
    """Fast path 1-2-5 (6 min) vs scenic loop 1-3-4-5 (12 min, high scenic_t)."""
    G = nx.MultiDiGraph()
    for u, v, tt_min, t, score in [
        (1, 2, 3.0, 0.2, 1.0), (2, 5, 3.0, 0.2, 1.0),     # fast, dull
        (1, 3, 4.0, 0.99, 3.0), (3, 4, 4.0, 0.99, 3.0),   # scenic loop
        (4, 5, 4.0, 0.99, 3.0),
    ]:
        G.add_edge(u, v, travel_time=tt_min / 60, scenic_t=t, scenic_score=score,
                   length=1000.0)
    for n in G.nodes:
        G.nodes[n]['y'] = 42.0 + n * 0.01
        G.nodes[n]['x'] = -76.0
    return G


def test_plan_scenic_route_takes_worthwhile_detour():
    G = _detour_graph()
    fast, scenic, fast_min, scenic_min = plan_scenic_route(G, 1, 5, 3.0)
    assert fast == [1, 2, 5]
    assert scenic == [1, 3, 4, 5]  # +6 min for +2.0 scenic gain: worth it
    assert scenic_min > fast_min


def test_plan_scenic_route_tight_slider_rejects_loop():
    # Tight slider (1.5): budget 6 + min(3, 12) = 9 min rejects the 12-min loop
    # (the default 3.0 admits it — covered by the worthwhile-detour test).
    G = _detour_graph()
    _, scenic, _, _ = plan_scenic_route(G, 1, 5, 1.5)
    assert scenic == [1, 2, 5]


def test_plan_scenic_route_skips_worthless_detour():
    # Loop barely more scenic: linear time penalty must reject it.
    G = _detour_graph()
    for u, v, k, d in G.edges(data=True, keys=True):
        if d['scenic_t'] > 0.5:
            d['scenic_score'] = 1.02  # gain 0.02 for 6 extra min: not worth it
    _, scenic, _, _ = plan_scenic_route(G, 1, 5, 3.0)
    assert scenic == [1, 2, 5]


def test_road_type_ref_rescue():
    assert _road_type_score({'highway': 'residential', 'ref': 'NY 55'}) == 0.8
    assert _road_type_score({'highway': 'residential'}) == 0.3
    assert _road_type_score({'highway': 'unclassified', 'ref': ['US 44', 'NY 55']}) == 0.8
    assert _road_type_score({'highway': 'unclassified'}) == 0.7


from route_planner import _trim_to_simple


def test_trim_to_simple_removes_out_and_back():
    # 1-2-3-2-4: out to 3 and back through 2 -> 1-2-4
    assert _trim_to_simple([1, 2, 3, 2, 4]) == [1, 2, 4]


def test_trim_to_simple_keeps_simple_path():
    assert _trim_to_simple([1, 2, 3, 4]) == [1, 2, 3, 4]


def test_trim_to_simple_nested_cycles():
    assert _trim_to_simple([1, 2, 3, 4, 3, 5, 2, 6]) == [1, 2, 6]


def test_road_type_ref_rescue_primary():
    # Scenic state highways tagged primary (NY-97, US-44/NY-55) get rescued...
    assert _road_type_score({'highway': 'primary', 'ref': 'US 44;NY 55'}) == 0.8
    assert _road_type_score({'highway': 'primary'}) == 0.3
    # ...but trunk/motorway never are.
    assert _road_type_score({'highway': 'trunk', 'ref': 'US 209'}) == 0.15


from shapely.geometry import LineString
from route_planner import _curviness_score


def _edge(coords_lonlat, length_m):
    return {'geometry': LineString(coords_lonlat), 'length': length_m}


def test_curviness_straight_road_is_zero():
    # 1 km due east at the equator: no bearing change.
    e = _edge([(0.0, 0.0), (0.005, 0.0), (0.009, 0.0)], 1000.0)
    assert _curviness_score(e) == 0.0


def test_curviness_hairpin_scores_high():
    # Out east 200m, hairpin, back west 200m: ~180 degrees over ~400m.
    e = _edge([(0.0, 0.0), (0.0018, 0.0), (0.0018, 0.0002), (0.0, 0.0002)], 400.0)
    assert _curviness_score(e) > 0.8


def test_curviness_gentle_curve_is_moderate():
    # ~45 degree total bend over ~700m.
    e = _edge([(0.0, 0.0), (0.003, 0.0), (0.006, 0.0021)], 700.0)
    score = _curviness_score(e)
    assert 0.1 < score < 0.7


def test_curviness_no_geometry_is_zero():
    assert _curviness_score({'length': 500.0}) == 0.0


def test_curviness_short_stub_is_zero():
    e = _edge([(0.0, 0.0), (0.00005, 0.00002)], 6.0)
    assert _curviness_score(e) == 0.0


from route_planner import _straight_miles, LONG_TRIP_MILES


def test_straight_miles_known_distance():
    # Ithaca to Columbus Circle NYC: ~174 miles straight-line.
    d = _straight_miles((42.454, -76.485), (40.768, -73.982))
    assert 165 < d < 185


def test_long_trip_threshold_classification():
    ithaca, nyc = (42.454, -76.485), (40.768, -73.982)
    taughannock = (42.547, -76.607)
    assert _straight_miles(ithaca, nyc) > LONG_TRIP_MILES
    assert _straight_miles(ithaca, taughannock) < LONG_TRIP_MILES


def test_budget_scales_with_trip_length():
    # 3-hour fast trip: slack = 20% = 36 min, so a +30-min detour with real
    # gain is admitted; on the 6-min synthetic trip the 12-min ceiling holds.
    G = nx.MultiDiGraph()
    for u, v, tt_min, t, score in [
        (1, 2, 90.0, 0.2, 1.0), (2, 5, 90.0, 0.2, 1.0),      # fast: 180 min
        (1, 3, 70.0, 0.99, 3.0), (3, 4, 70.0, 0.99, 3.0),    # scenic: 210 min
        (4, 5, 70.0, 0.99, 3.0),
    ]:
        G.add_edge(u, v, travel_time=tt_min / 60, scenic_t=t, scenic_score=score,
                   length=1000.0)
    for node in G.nodes:
        G.nodes[node]['y'] = 42.0 + node * 0.01
        G.nodes[node]['x'] = -76.0
    _, scenic, fast_min, scenic_min = plan_scenic_route(G, 1, 5, 3.0)
    assert scenic == [1, 3, 4, 5]  # +30 min on a 3-hour trip: within 20% slack


from route_planner import _scenic_anchors


def test_scenic_anchors_exclude_endpoint_clusters():
    # A top-scenic cluster at the destination must not take an anchor slot:
    # scenery at an endpoint needs no detour (Ithaca->NYC: Manhattan's
    # viewpoint clusters claimed all four anchors and every candidate
    # collapsed onto the fast route).
    G = nx.MultiDiGraph()
    coords = {1: 42.0, 2: 43.0, 10: 42.02, 11: 42.03, 20: 42.5, 21: 42.51}
    for u, v, t in [(1, 2, 0.2), (10, 11, 0.995), (20, 21, 0.995)]:
        G.add_edge(u, v, travel_time=0.1, scenic_t=t, scenic_score=1.0,
                   length=1000.0)
    for node, lat in coords.items():
        G.nodes[node]['y'] = lat
        G.nodes[node]['x'] = -76.0
    # Trip 1 -> 2 spans 1 deg; exclusion radius 0.15 deg swallows the cluster
    # near the start but not the mid-route one.
    anchors = _scenic_anchors(G, 1, 2)
    assert 21 in anchors
    assert 11 not in anchors
    # Without trip context both clusters are eligible.
    assert set(_scenic_anchors(G)) == {11, 21}


from route_planner import _parse_maxspeed_mph


def test_maxspeed_explicit_mph():
    assert _parse_maxspeed_mph("35 mph") == 35.0


def test_maxspeed_bare_number_is_kmh():
    # OSM default unit is km/h: 80 km/h ~= 49.7 mph.
    assert abs(_parse_maxspeed_mph("80") - 49.7) < 0.1
    assert abs(_parse_maxspeed_mph(80) - 49.7) < 0.1


def test_maxspeed_list_takes_first():
    assert _parse_maxspeed_mph(["50 mph", "40 mph"]) == 50.0


def test_maxspeed_unparseable_is_none():
    assert _parse_maxspeed_mph("none") is None
    assert _parse_maxspeed_mph(None) is None


def test_maxspeed_nan_is_none():
    # pyrosm emits NaN (not None) for untagged maxspeed; NaN must not leak
    # into travel_time (it slips past the caller's `not speed` guard).
    assert _parse_maxspeed_mph(float("nan")) is None


def test_ref_rescue_norwegian_county_roads():
    assert _road_type_score({'highway': 'secondary', 'ref': 'Fv 815'}) == 0.8


def test_ref_rescue_denied_to_multilane_arterials():
    # NY 59 through Rockland County: state ref, but 4 lanes of strip mall.
    assert _road_type_score({'highway': 'primary', 'ref': 'NY 59', 'lanes': '4'}) == 0.3
    # Two-laners (and untagged lanes) keep the rescue.
    assert _road_type_score({'highway': 'primary', 'ref': 'NY 97', 'lanes': '2'}) == 0.8
    assert _road_type_score({'highway': 'primary', 'ref': 'Rv 15'}) == 0.8


def test_ref_rescue_excludes_european_e_roads():
    # E10 is a trunk-network designation, not a scenic road.
    assert _road_type_score({'highway': 'primary', 'ref': 'E 10'}) == 0.3


def test_ref_rescue_word_boundary():
    # 'NE 2' must not match the E pattern (or any other).
    assert _road_type_score({'highway': 'primary', 'ref': 'NE 2'}) == 0.3


def test_scenic_yes_tag_boosts_any_class():
    assert _road_type_score({'highway': 'motorway', 'scenic': 'yes'}) == 0.9
    assert _road_type_score({'highway': 'residential', 'scenic': 'yes'}) == 0.9


def test_ref_rescue_bare_numeric_county_roads():
    # Norwegian fylkesvei carry bare numeric refs (ref=815, class primary).
    assert _road_type_score({'highway': 'primary', 'ref': '815'}) == 0.8
    # E-roads still excluded ('E 10' is not bare digits).
    assert _road_type_score({'highway': 'trunk', 'ref': 'E 10'}) == 0.15


from route_planner import bbox_for


def test_bbox_lon_buffer_scales_with_latitude():
    # At 68N, 0.1 deg of lon is ~4.5 km; the buffer must widen to stay ~11 km.
    n, s, e, w = bbox_for((68.15, 14.20), (68.08, 13.53))
    assert (e - 14.20) > 0.25
    # At the equator lon and lat buffers match.
    n2, s2, e2, w2 = bbox_for((0.0, 10.0), (0.5, 10.5))
    assert abs((e2 - 10.5) - 0.1) < 1e-4


def test_marginal_gain_stays_on_fast_route():
    # Value below the dead-band (0.05): a tiny wiggle must not replace fast.
    G = _detour_graph()
    for u, v, k, d in G.edges(data=True, keys=True):
        if d['scenic_t'] > 0.5:
            d['scenic_score'] = 1.65  # gain 0.65 - penalty 0.5*(6/6) = 0.15 > band: takes it
    _, scenic, _, _ = plan_scenic_route(G, 1, 5, 3.0)
    assert scenic == [1, 3, 4, 5]
    for u, v, k, d in G.edges(data=True, keys=True):
        if d['scenic_t'] > 0.5:
            d['scenic_score'] = 1.50  # gain 0.5 - 0.5 = 0.0 < band: stays fast
    _, scenic, _, _ = plan_scenic_route(G, 1, 5, 3.0)
    assert scenic == [1, 2, 5]
