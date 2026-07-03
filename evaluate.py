"""Scenic-route validation harness. Runs every fixture through the routing
pipeline and prints a scorecard.

Usage: python evaluate.py [name-substring]
An optional argument runs only fixtures whose name contains it (useful when
one fixture needs a long first-time graph fetch).

Headline columns: cover (waypoint coverage), gain (time-weighted mean scenic
score, scenic minus fast), xmin (extra minutes spent). Diagnostics: detour
(must stay within each fixture's cap), ndiv/ldiv (divergence — high divergence
with low gain means the route is wandering, not scenic).
"""
import json
import os
import sys

import osmnx
from geopy.geocoders import Nominatim

from route_planner import initialize_graph, score_scenic_edges, plan_scenic_route, route_coords
from evaluation.fixtures import FIXTURES
from evaluation.metrics import (
    waypoint_coverage, shared_node_fraction, shared_length_fraction,
    detour_ratio, mean_scenic_score, geo_dist_m,
)

_GEO = Nominatim(user_agent="ramble_eval")
_CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".geocode_cache.json")


def _load_cache():
    if os.path.exists(_CACHE_PATH):
        with open(_CACHE_PATH) as f:
            return json.load(f)
    return {}


def _save_cache(cache):
    with open(_CACHE_PATH, "w") as f:
        json.dump(cache, f, indent=2)


def geocode(name, cache):
    if name in cache:
        return tuple(cache[name])
    loc = _GEO.geocode(name)
    if loc is None:
        raise ValueError(f"geocode failed: {name}")
    point = (loc.latitude, loc.longitude)
    cache[name] = list(point)
    return point


def run_fixture(fixture, cache, weights):
    start = geocode(fixture["start"], cache)
    end = geocode(fixture["end"], cache)
    # via entries: (name, threshold_m) geocoded by name, or
    # (label, threshold_m, (lat, lon)) with an explicit reference point for
    # places whose names geocode badly.
    via = [(entry[2] if len(entry) > 2 else geocode(entry[0], cache), entry[1])
           for entry in fixture.get("via", [])]

    G, _ = initialize_graph(start, end)
    start_node = osmnx.distance.nearest_nodes(G, X=[start[1]], Y=[start[0]])[0]
    end_node = osmnx.distance.nearest_nodes(G, X=[end[1]], Y=[end[0]])[0]

    # Snap guard: a via point that geocodes far from any road (e.g. a park
    # interior) can never be covered — fail loudly instead of reporting a
    # misleading 0.00 coverage.
    for (point, threshold), (name, *_rest) in zip(via, fixture.get("via", [])):
        node = osmnx.distance.nearest_nodes(G, X=[point[1]], Y=[point[0]])[0]
        snap_m = geo_dist_m(point, (G.nodes[node]['y'], G.nodes[node]['x']))
        if snap_m > threshold:
            raise ValueError(
                f"via '{name}' lands {snap_m:.0f} m from the road network "
                f"(> {threshold:.0f} m threshold) — bad reference point")

    score_scenic_edges(G, weights)
    fast, scenic, fast_min, scenic_min = plan_scenic_route(
        G, start_node, end_node, fixture.get("max_detour", 3.0)
    )
    polyline = route_coords(G, scenic)

    return {
        "name": fixture["name"],
        "coverage": waypoint_coverage(via, polyline),
        "gain": mean_scenic_score(G, scenic) - mean_scenic_score(G, fast),
        "xmin": scenic_min - fast_min,
        "detour": detour_ratio(fast_min, scenic_min),
        "ndiv": 1.0 - shared_node_fraction(fast, scenic),
        "ldiv": 1.0 - shared_length_fraction(G, fast, scenic),
    }


def main():
    cache = _load_cache()
    weights = {"curviness": 1.0, "road_type": 1.0, "nature": 1.0}
    name_filter = sys.argv[1].lower() if len(sys.argv) > 1 else ""
    rows = []
    for fixture in FIXTURES:
        if name_filter and name_filter not in fixture["name"].lower():
            continue
        try:
            rows.append(run_fixture(fixture, cache, weights))
        except Exception as exc:  # report, don't abort the whole run
            rows.append({"name": fixture["name"], "error": str(exc)})
        _save_cache(cache)

    header = f"{'fixture':38} {'cover':>6} {'gain':>7} {'xmin':>6} {'detour':>7} {'ndiv':>6} {'ldiv':>6}"
    print(header)
    print("-" * len(header))
    for r in rows:
        if "error" in r:
            print(f"{r['name'][:38]:38} ERROR: {r['error']}")
            continue
        cover = "-" if r["coverage"] is None else f"{r['coverage']:.2f}"
        print(f"{r['name'][:38]:38} {cover:>6} {r['gain']:>+7.2f} {r['xmin']:>6.1f} "
              f"{r['detour']:>7.2f} {r['ndiv']:>6.2f} {r['ldiv']:>6.2f}")


if __name__ == "__main__":
    main()
