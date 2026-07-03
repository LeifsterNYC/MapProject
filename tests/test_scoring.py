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


def test_plan_scenic_route_additive_slack_beats_tight_multiplier():
    # Cap 1.5x = 9 min < loop's 12 min, but additive slack (+15 min) admits it.
    G = _detour_graph()
    _, scenic, _, _ = plan_scenic_route(G, 1, 5, 1.5)
    assert scenic == [1, 3, 4, 5]


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
