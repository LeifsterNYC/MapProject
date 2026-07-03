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
