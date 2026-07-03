# Scenic Route Quality Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Ramble's scenic route genuinely diverge from the fastest route, and make "is it right?" measurable via a validation harness.

**Architecture:** Build a validation harness (fixtures → runner → scorecard) that measures scenic-route quality, capture a baseline, then rework `score_scenic_edges` so proximity signals discriminate (relative/percentile-based, not absolute-radius) and validate each change against the scorecard.

> **AMENDED 2026-07-03 after prototype review** (see spec Amendments A–E): fixtures
> are real user-supplied trips (Task 7); waypoint coverage takes per-via thresholds
> (Task 3); the scorecard leads with coverage + scenic gain per extra minute, with
> divergence demoted to diagnostic (Task 8); Task 10 uses empirical percentile
> normalization of the total score (`scenic_t`), not divide-by-theoretical-max;
> a new Task 10b replaces fixed `_SCENIC_K` + cliff fallback with an adaptive
> K-ladder search in `plan_scenic_route`; Task 11 adds an anti-wandering gate.
> Where a task body below conflicts with these amendments, the amendment wins.

**Tech Stack:** Python 3.11, Flask, OSMnx 2.1, NetworkX 3.6, scipy KDTree, numpy, pytest.

---

## Environment

All commands assume the `mapproject` conda env is active:

```bash
conda activate mapproject
cd /Users/leifrogers/Documents/dev/MapProject
```

`python` and `pytest` below refer to that env's interpreter
(`/opt/homebrew/Caskroom/miniforge/base/envs/mapproject/bin/python`).

## File Structure

- `tests/` — **new** — pytest unit tests (no network access)
- `evaluation/__init__.py` — **new** — package marker
- `evaluation/fixtures.py` — **new** — list of test trips (start/end/via/max_detour)
- `evaluation/metrics.py` — **new** — pure metric functions (unit-tested)
- `evaluate.py` — **new** — harness runner + scorecard (network + cached graphs)
- `route_planner.py` — **modified** — add `route_coords`, add `relative_proximity_scores`, rework `score_scenic_edges`, tighten road-type/traffic tables
- `app.py` — **modified** — import `route_coords` from `route_planner` instead of its local `_route_coords`
- `CLAUDE.md` — **modified** — refresh stale architecture notes

---

### Task 1: Test infrastructure

**Files:**
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Create: `evaluation/__init__.py`

- [ ] **Step 1: Install pytest into the env**

Run: `python -m pip install pytest`
Expected: installs pytest successfully.

- [ ] **Step 2: Create package/test marker files**

Create `tests/__init__.py` (empty file).
Create `evaluation/__init__.py` (empty file).
Create `tests/conftest.py` with:

```python
# Ensures the project root is importable when running pytest from anywhere.
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
```

- [ ] **Step 3: Verify pytest collects nothing yet (sanity)**

Run: `python -m pytest -q`
Expected: `no tests ran` (exit code 5) — confirms pytest is installed and importable.

- [ ] **Step 4: Commit**

```bash
git add tests/__init__.py tests/conftest.py evaluation/__init__.py
git commit -m "test: add pytest infrastructure"
```

---

### Task 2: Geometry helpers in metrics.py (`geo_dist_m`, `point_to_polyline_m`)

**Files:**
- Create: `evaluation/metrics.py`
- Test: `tests/test_metrics.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_metrics.py`:

```python
from evaluation.metrics import geo_dist_m, point_to_polyline_m


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_metrics.py -q`
Expected: FAIL — `ModuleNotFoundError` / `ImportError: cannot import name 'geo_dist_m'`.

- [ ] **Step 3: Write minimal implementation**

Create `evaluation/metrics.py`:

```python
"""Pure metric functions for scoring scenic-route quality. No network access."""
import math


def geo_dist_m(a, b):
    """Haversine distance in metres between two (lat, lon) points."""
    lat1, lon1 = a
    lat2, lon2 = b
    R = 6_371_008.8  # mean Earth radius (m)
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    h = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlam / 2) ** 2
    return 2 * R * math.asin(math.sqrt(h))


def point_to_polyline_m(point, polyline):
    """Minimum distance (m) from a (lat, lon) point to a polyline's vertices.

    Route polylines follow dense road geometry, so nearest-vertex is a close
    approximation of nearest-point and keeps this dependency-free and testable.
    """
    return min(geo_dist_m(point, vertex) for vertex in polyline)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_metrics.py -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add evaluation/metrics.py tests/test_metrics.py
git commit -m "feat: add geo_dist_m and point_to_polyline_m metrics"
```

---

### Task 3: Waypoint coverage metric

**Files:**
- Modify: `evaluation/metrics.py`
- Test: `tests/test_metrics.py`

- [ ] **Step 1: Add the failing tests**

Append to `tests/test_metrics.py`:

```python
from evaluation.metrics import waypoint_coverage


def test_waypoint_coverage_all_hit():
    via = [(0.0, 0.0), (0.0, 1.0)]
    polyline = [(0.0, 0.0), (0.0, 0.5), (0.0, 1.0)]
    assert waypoint_coverage(via, polyline, threshold_m=300) == 1.0


def test_waypoint_coverage_half_hit():
    via = [(0.0, 0.0), (10.0, 10.0)]  # second is ~1500 km away
    polyline = [(0.0, 0.0), (0.0, 0.001)]
    assert waypoint_coverage(via, polyline, threshold_m=300) == 0.5


def test_waypoint_coverage_empty_via_is_none():
    assert waypoint_coverage([], [(0.0, 0.0)]) is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_metrics.py -q`
Expected: FAIL — `ImportError: cannot import name 'waypoint_coverage'`.

- [ ] **Step 3: Add the implementation**

Append to `evaluation/metrics.py`:

```python
def waypoint_coverage(via_points, polyline, threshold_m=300.0):
    """Fraction of via_points within threshold_m of the polyline.

    Returns None when there are no via points (no reference to score against).
    """
    if not via_points:
        return None
    hits = sum(1 for p in via_points if point_to_polyline_m(p, polyline) <= threshold_m)
    return hits / len(via_points)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_metrics.py -q`
Expected: PASS (6 passed).

- [ ] **Step 5: Commit**

```bash
git add evaluation/metrics.py tests/test_metrics.py
git commit -m "feat: add waypoint_coverage metric"
```

---

### Task 4: Divergence and detour metrics

**Files:**
- Modify: `evaluation/metrics.py`
- Test: `tests/test_metrics.py`

- [ ] **Step 1: Add the failing tests**

Append to `tests/test_metrics.py`:

```python
from evaluation.metrics import shared_node_fraction, detour_ratio


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_metrics.py -q`
Expected: FAIL — `ImportError: cannot import name 'shared_node_fraction'`.

- [ ] **Step 3: Add the implementation**

Append to `evaluation/metrics.py`:

```python
def shared_node_fraction(fast_nodes, scenic_nodes):
    """Jaccard overlap of the two node sets (1.0 = identical, 0.0 = disjoint)."""
    if not scenic_nodes:
        return 0.0
    union = set(fast_nodes) | set(scenic_nodes)
    if not union:
        return 0.0
    shared = set(fast_nodes) & set(scenic_nodes)
    return len(shared) / len(union)


def detour_ratio(fast_time, scenic_time):
    """Scenic travel time divided by fast travel time (same units)."""
    if fast_time <= 0:
        return 1.0
    return scenic_time / fast_time
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_metrics.py -q`
Expected: PASS (12 passed).

- [ ] **Step 5: Commit**

```bash
git add evaluation/metrics.py tests/test_metrics.py
git commit -m "feat: add divergence and detour metrics"
```

---

### Task 5: Graph-based metrics (`shared_length_fraction`, `mean_scenic_score`)

**Files:**
- Modify: `evaluation/metrics.py`
- Test: `tests/test_metrics.py`

- [ ] **Step 1: Add the failing tests**

Append to `tests/test_metrics.py`:

```python
import networkx as nx
from evaluation.metrics import shared_length_fraction, mean_scenic_score


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
    fast = [1, 2, 4]      # edges (1,2)=100, (2,4)=300  -> 400 total
    scenic = [1, 2, 3, 4]  # edges (1,2)=100 shared, (2,3)=100, (3,4)=100 -> 300 total
    # shared length along scenic = 100 (edge 1-2); total scenic length = 300
    assert shared_length_fraction(G, fast, scenic) == 100.0 / 300.0


def test_mean_scenic_score():
    G = _tiny_graph()
    scenic = [1, 2, 3, 4]  # scores 0.2, 0.8, 0.8
    assert abs(mean_scenic_score(G, scenic) - 0.6) < 1e-9
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_metrics.py -q`
Expected: FAIL — `ImportError: cannot import name 'shared_length_fraction'`.

- [ ] **Step 3: Add the implementation**

Append to `evaluation/metrics.py`:

```python
def _path_edges(G, path):
    """Yield (u, v, data) for the min-travel_time parallel edge of each hop."""
    for u, v in zip(path[:-1], path[1:]):
        data = min(G[u][v].values(), key=lambda d: d.get('travel_time', 0))
        yield u, v, data


def shared_length_fraction(G, fast, scenic):
    """Fraction of the scenic route's length (m) that reuses fast-route edges."""
    fast_edges = {frozenset((u, v)) for u, v, _ in _path_edges(G, fast)}
    total = 0.0
    shared = 0.0
    for u, v, data in _path_edges(G, scenic):
        length = data.get('length', 0.0)
        total += length
        if frozenset((u, v)) in fast_edges:
            shared += length
    return shared / total if total else 0.0


def mean_scenic_score(G, path):
    """Mean edge scenic_score along a path."""
    scores = [data.get('scenic_score', 0.0) for _, _, data in _path_edges(G, path)]
    return sum(scores) / len(scores) if scores else 0.0
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_metrics.py -q`
Expected: PASS (14 passed).

- [ ] **Step 5: Commit**

```bash
git add evaluation/metrics.py tests/test_metrics.py
git commit -m "feat: add graph-based length and scenic-score metrics"
```

---

### Task 6: Share `route_coords` between app and harness

**Files:**
- Modify: `route_planner.py` (add public `route_coords`)
- Modify: `app.py:44-60` (remove `_route_coords`, import from `route_planner`)

- [ ] **Step 1: Add `route_coords` to route_planner.py**

Append this function to `route_planner.py` (it is the exact body currently in
`app.py._route_coords`, made public):

```python
def route_coords(G, route):
    """Extract (lat, lon) coords following actual road geometry, not just nodes."""
    coords = []
    for i in range(len(route) - 1):
        u, v = route[i], route[i + 1]
        edge_data = min(G[u][v].values(), key=lambda d: d.get('travel_time', 0))
        geom = edge_data.get('geometry')
        if geom is not None:
            edge_coords = [(y, x) for x, y in geom.coords]
        else:
            edge_coords = [(G.nodes[u]['y'], G.nodes[u]['x']),
                           (G.nodes[v]['y'], G.nodes[v]['x'])]
        coords.extend(edge_coords if not coords else edge_coords[1:])
    if not coords and route:
        coords.append((G.nodes[route[0]]['y'], G.nodes[route[0]]['x']))
    return coords
```

- [ ] **Step 2: Update app.py to import it**

In `app.py`, change the import line (currently line 16):

```python
from route_planner import initialize_graph, score_scenic_edges, plan_scenic_route
```

to:

```python
from route_planner import initialize_graph, score_scenic_edges, plan_scenic_route, route_coords
```

Delete the entire `_route_coords` function definition from `app.py` (lines 44-60).

In `app.py._pipeline`, replace the two calls (currently lines 180-181):

```python
    fast_coords = _route_coords(graph, fast_route)
    scenic_coords = _route_coords(graph, scenic_route)
```

with:

```python
    fast_coords = route_coords(graph, fast_route)
    scenic_coords = route_coords(graph, scenic_route)
```

- [ ] **Step 3: Verify app still imports and route_coords is shared**

Run: `python -c "import app; from route_planner import route_coords; print('OK', app._pipeline is not None)"`
Expected: prints `OK True` with no `NameError` for `_route_coords`.

- [ ] **Step 4: Commit**

```bash
git add app.py route_planner.py
git commit -m "refactor: share route_coords between app and harness"
```

---

### Task 7: Fixtures

**Files:**
- Create: `evaluation/fixtures.py`

- [ ] **Step 1: Create the fixture list**

Create `evaluation/fixtures.py`:

```python
"""Test trips for the scenic-route validation harness.

Each fixture:
  name       - human label for the scorecard
  start/end  - addresses (geocoded via Nominatim)
  via        - place-name strings the scenic route SHOULD pass near
               (captured from local knowledge or an online scenic detour);
               geocoded to points and checked with waypoint_coverage
  max_detour - allowed scenic_time / fast_time ratio for this trip

Choosing via names: prefer TOWNS, PARKS, and LANDMARKS - they geocode to tight
points. Avoid bare long-road names ("Taughannock Blvd"): Nominatim resolves a
road to ONE arbitrary point on it, which can be miles from where the route uses
the road, producing false coverage failures. If a road is the reference, anchor
it with a cross-street or landmark ("Taughannock Blvd at Glenwood Rd").

Add 5-8 real trips. The first is the app's default route as a smoke case.
"""

FIXTURES = [
    # via entries are (name, threshold_m): road names geocode to one arbitrary
    # point, so they get looser thresholds than towns/landmarks.
    {
        "name": "TJ's Ithaca -> Dryden Rd (Sand Bank curves)",
        "start": "Trader Joe's, Ithaca, NY",
        "end": "205 Dryden Rd, Ithaca, NY 14850",
        "via": [("Sand Bank Road, Ithaca, NY", 1200),
                ("West King Road, Ithaca, NY", 1200)],
        "max_detour": 3.0,
    },
    {
        "name": "Accord -> New Paltz (Gunks via 44/55)",
        "start": "351 Cooper St, Accord, NY 12404",
        "end": "New Paltz, NY",
        "via": [("Kerhonkson, NY", 1200),
                ("Minnewaska State Park Preserve", 2500)],
        "max_detour": 3.0,
    },
    {
        "name": "Hancock -> Port Jervis (Rt 97 Delaware)",
        "start": "Hancock, NY",
        "end": "Port Jervis, NY",
        "via": [("Narrowsburg, NY", 1500),
                ("Barryville, NY", 1500)],
        "max_detour": 3.0,
    },
    {
        # Smoke case: fast route (Rt 89 lakeside) already IS scenic here.
        # Expected: little/no divergence and NO city-street wandering.
        "name": "Ithaca -> Taughannock (smoke: fast==scenic)",
        "start": "312 Thurston Ave, Ithaca, NY 14850",
        "end": "1781 Taughannock Blvd, Ulysses, NY 14886",
        "via": [("Taughannock Falls State Park, NY", 1500)],
        "max_detour": 3.0,
    },
]
```

> Future (out of scope, exceeds 50-mile scale): Ithaca -> NYC preferring NY-17 +
> Palisades over I-380/I-80.

- [ ] **Step 2: Verify it imports**

Run: `python -c "from evaluation.fixtures import FIXTURES; print(len(FIXTURES), 'fixtures')"`
Expected: prints `1 fixtures`.

- [ ] **Step 3: Commit**

```bash
git add evaluation/fixtures.py
git commit -m "feat: add scenic-route test fixtures"
```

---

### Task 8: Harness runner + baseline scorecard

**Files:**
- Create: `evaluate.py`
- Modify: `.gitignore` (ignore the geocode cache)

- [ ] **Step 1: Create evaluate.py**

Create `evaluate.py`:

```python
"""Scenic-route validation harness. Runs every fixture through the routing
pipeline and prints a scorecard. Usage: python evaluate.py"""
import json
import os

import osmnx
from geopy.geocoders import Nominatim

from route_planner import initialize_graph, score_scenic_edges, plan_scenic_route, route_coords
from evaluation.fixtures import FIXTURES
from evaluation.metrics import (
    waypoint_coverage, shared_node_fraction, shared_length_fraction,
    detour_ratio, mean_scenic_score,
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
    via = [geocode(n, cache) for n in fixture.get("via", [])]

    G, _ = initialize_graph(start, end)
    start_node = osmnx.distance.nearest_nodes(G, X=[start[1]], Y=[start[0]])[0]
    end_node = osmnx.distance.nearest_nodes(G, X=[end[1]], Y=[end[0]])[0]

    score_scenic_edges(G, weights)
    fast, scenic, fast_min, scenic_min = plan_scenic_route(
        G, start_node, end_node, fixture.get("max_detour", 3.0)
    )
    polyline = route_coords(G, scenic)

    return {
        "name": fixture["name"],
        "coverage": waypoint_coverage(via, polyline) if via else None,
        "node_div": 1.0 - shared_node_fraction(fast, scenic),
        "len_div": 1.0 - shared_length_fraction(G, fast, scenic),
        "detour": detour_ratio(fast_min, scenic_min),
        "gain": mean_scenic_score(G, scenic) - mean_scenic_score(G, fast),
    }


def main():
    cache = _load_cache()
    weights = {"curviness": 1.0, "road_type": 1.0, "nature": 1.0}
    rows = []
    for fixture in FIXTURES:
        try:
            rows.append(run_fixture(fixture, cache, weights))
        except Exception as exc:  # report, don't abort the whole run
            rows.append({"name": fixture["name"], "error": str(exc)})
    _save_cache(cache)

    header = f"{'fixture':34} {'cover':>6} {'ndiv':>6} {'ldiv':>6} {'detour':>7} {'gain':>7}"
    print(header)
    print("-" * len(header))
    for r in rows:
        if "error" in r:
            print(f"{r['name'][:34]:34} ERROR: {r['error']}")
            continue
        cover = "-" if r["coverage"] is None else f"{r['coverage']:.2f}"
        print(f"{r['name'][:34]:34} {cover:>6} {r['node_div']:>6.2f} "
              f"{r['len_div']:>6.2f} {r['detour']:>7.2f} {r['gain']:>+7.2f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Ignore the geocode cache**

Append to `.gitignore`:

```
.geocode_cache.json
```

- [ ] **Step 3: Run the harness to capture the BASELINE (pre-fix) scorecard**

Run: `python evaluate.py`
Expected: prints a scorecard. The seed fixture should show a **low `ndiv`/`ldiv`**
(scenic ~= fast — the bug we are fixing), e.g. `ndiv` well under 0.20. Record this
output; it is the "before" number.

> If geocoding or graph fetch is slow on first run, that is expected (network +
> ~114 MB cached graph). Subsequent runs use the geocode cache and in-memory graph.

- [ ] **Step 4: Commit**

```bash
git add evaluate.py .gitignore
git commit -m "feat: add scenic-route validation harness"
```

---

### Task 9: `relative_proximity_scores` helper

**Files:**
- Modify: `route_planner.py`
- Test: `tests/test_scoring.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_scoring.py`:

```python
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


def test_relative_proximity_empty_returns_empty():
    scores = relative_proximity_scores([])
    assert len(scores) == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_scoring.py -q`
Expected: FAIL — `ImportError: cannot import name 'relative_proximity_scores'`.

- [ ] **Step 3: Add the implementation**

Add to `route_planner.py` (near the other scoring helpers, after `_make_tree`):

```python
def relative_proximity_scores(distances, far_gate_m=2000.0):
    """Convert per-edge distances-to-a-feature into a discriminating [0, 1] score.

    Percentile-rank based: the closest edge scores 1.0, the farthest 0.0, spread
    evenly in between — so the signal separates edges *within this graph* instead
    of saturating near 1.0 in feature-rich regions (the root-cause bug).

    If the feature is effectively absent (median distance beyond far_gate_m, or an
    empty/all-infinite distance array), every edge scores 0.0 so a missing feature
    contributes nothing rather than rewarding merely-least-far edges.
    """
    d = np.asarray(distances, dtype=float)
    n = d.size
    if n == 0:
        return d
    finite = d[np.isfinite(d)]
    if finite.size == 0 or np.median(finite) > far_gate_m:
        return np.zeros(n)
    order = d.argsort()
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.arange(n)
    return 1.0 - ranks / max(n - 1, 1)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_scoring.py -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add route_planner.py tests/test_scoring.py
git commit -m "feat: add relative_proximity_scores for discriminating scenic signals"
```

---

### Task 10: Rework `score_scenic_edges` to use relative proximity

**Files:**
- Modify: `route_planner.py` (replace `score_scenic_edges` body, add `_SCENIC_K` constant)

This is a quality change validated by the harness scorecard, not a unit test.

- [ ] **Step 1: Add the shaping constant**

Near the top of `route_planner.py`, below `_SIGNAL_DELAY_HR`, add:

```python
_SCENIC_K = 3.0  # scenic_cost = travel_time * exp(_SCENIC_K * (1 - normalized_score))
```

- [ ] **Step 2: Replace `score_scenic_edges`**

Replace the entire existing `score_scenic_edges` function with:

```python
def score_scenic_edges(G, weights: dict) -> None:
    w_curve = weights.get('curviness', 1.0)
    w_road = weights.get('road_type', 1.0)
    w_nature = weights.get('nature', 1.0)
    # Fixed-weight components.
    W_SPEED, W_TRAFFIC, W_WATER, W_VIEW = 1.0, 0.5, 1.0, 0.5
    max_score = w_curve + w_road + w_nature + W_SPEED + W_TRAFFIC + W_WATER + W_VIEW

    water_points = G.graph.get('water_points', [])
    view_points = G.graph.get('viewpoints', [])
    nature_points = G.graph.get('nature_points', [])

    ref_lat = sum(d.get('y', 0) for _, d in list(G.nodes(data=True))[:200]) / 200
    lat_m = 111320.0
    lon_m = 111320.0 * math.cos(math.radians(ref_lat))
    water_tree = _make_tree(water_points, lat_m, lon_m)
    view_tree = _make_tree(view_points, lat_m, lon_m)
    nature_tree = _make_tree(nature_points, lat_m, lon_m)

    edges = list(G.edges(data=True, keys=True))

    # Edge midpoints in (lat, lon).
    mids = []
    for u, v, k, data in edges:
        geom = data.get('geometry')
        if geom is not None:
            coords = list(geom.coords)
            m = coords[len(coords) // 2]
            mids.append((m[1], m[0]))
        else:
            mids.append(((G.nodes[u]['y'] + G.nodes[v]['y']) / 2,
                         (G.nodes[u]['x'] + G.nodes[v]['x']) / 2))

    def dists_to(tree):
        if tree is None:
            return np.full(len(mids), np.inf)
        q = np.array([(lat * lat_m, lon * lon_m) for lat, lon in mids])
        d, _ = tree.query(q)
        return d

    # Dense features (water, nature: thousands of points) use relative
    # (percentile-rank) proximity so they discriminate within this graph instead
    # of saturating near 1.0 in feature-rich regions.
    # Sparse features (viewpoints: often <10 in a bbox) keep an ABSOLUTE 200 m
    # gradient - percentile ranking would gate them to zero everywhere because
    # the median edge is legitimately far from any viewpoint.
    water_rel = relative_proximity_scores(dists_to(water_tree))
    nature_rel = relative_proximity_scores(dists_to(nature_tree))
    view_abs = np.maximum(0.0, 1.0 - dists_to(view_tree) / 200.0)

    for i, (u, v, k, data) in enumerate(edges):
        scenic_score = (
            w_curve * _curviness_score(data) +
            w_road * _road_type_score(data) +
            w_nature * float(nature_rel[i]) +
            W_SPEED * _speed_score(data) +
            W_TRAFFIC * _traffic_score(data) +
            W_WATER * float(water_rel[i]) +
            W_VIEW * float(view_abs[i])
        )
        data['scenic_score'] = scenic_score
        t = min(scenic_score / max_score, 1.0) if max_score > 0 else 0.0
        data['scenic_cost'] = data['travel_time'] * math.exp(_SCENIC_K * (1.0 - t))
```

- [ ] **Step 3: Confirm metric unit tests still pass**

Run: `python -m pytest -q`
Expected: PASS (all metric + scoring unit tests green; scoring change has no unit test but must not break imports).

- [ ] **Step 4: Run the harness and compare to baseline**

Run: `python evaluate.py`
Expected: on the seed fixture, `ndiv` and `ldiv` are **substantially higher** than
the Task 8 baseline (scenic route now diverges from fast), while `detour` stays
within the fixture's `max_detour`. Record the new scorecard.

> If divergence is still low, that is a signal to tune — proceed to Task 11 and,
> if needed, adjust `_SCENIC_K` (higher = stronger pull toward scenic edges) and
> re-run the harness. Do not accept the change until divergence clearly improves.

- [ ] **Step 5: Commit**

```bash
git add route_planner.py
git commit -m "feat: rework scenic scoring to use relative proximity signals"
```

---

### Task 10b (NEW, per Amendment D): Adaptive detour search in `plan_scenic_route`

**Files:**
- Modify: `route_planner.py` (`score_scenic_edges` stores `scenic_t` per edge instead
  of baking a fixed-K `scenic_cost`; `plan_scenic_route` gains the K-ladder search)

Behavior is knife-edged in K, so no fixed K works across trips. Instead:

- [ ] `score_scenic_edges` writes `scenic_score` and percentile-ranked `scenic_t`
      (empirical normalization, Amendment C) on every edge; no `scenic_cost`.
- [ ] `plan_scenic_route` probes K in `[1, 2, 3, 4, 5, 6, 8]`: for each K, set
      `scenic_cost = travel_time * exp(K * (1 - scenic_t))` and run Dijkstra.
      Deduplicate identical routes; drop candidates over `max_detour`.
- [ ] Among surviving candidates, return the one maximizing scenic gain per extra
      minute: `(tw_score(scenic) - tw_score(fast)) / max(extra_minutes, 1.0)` where
      `tw_score` is the travel-time-weighted mean `scenic_score`. The fast route is
      the floor candidate (gain 0), so the cliff fallback disappears naturally.
- [ ] Validate with the harness: coverage/gain up, smoke fixture does not wander.

---

### Task 11: Tighten road-type and traffic heuristics

**Files:**
- Modify: `route_planner.py` (`_road_type_score`, `_traffic_score`)

Quality change validated by the harness scorecard.

- [ ] **Step 1: Update the road-type table**

Replace the `table` dict inside `_road_type_score` with (lowers `unclassified`
below pleasant country roads and de-prioritizes `service` tracks so routes stop
favoring low-quality ways):

```python
    table = {
        'motorway': 0.15, 'trunk': 0.15, 'primary': 0.3,
        'motorway_link': 0.0, 'trunk_link': 0.0, 'primary_link': 0.1,
        'secondary': 0.5, 'secondary_link': 0.5,
        'tertiary': 0.9, 'tertiary_link': 0.9,
        'unclassified': 0.7, 'residential': 0.3,
        'living_street': 0.5, 'service': 0.05,
    }
```

- [ ] **Step 2: Reduce the no-lane-tag default in `_traffic_score`**

In `_traffic_score`, change the no-lanes branch (currently `return 0.4`):

```python
    if lanes is None:
        return 0.4  # no lanes tag -> quiet single-lane country road
```

to:

```python
    if lanes is None:
        return 0.3  # no lanes tag -> probably quiet, but do not over-reward
```

- [ ] **Step 3: Run unit tests and the harness**

Run: `python -m pytest -q`
Expected: PASS (unchanged — these functions have no unit tests; imports must work).

Run: `python evaluate.py`
Expected: scorecard still shows healthy `ndiv`/`ldiv` with `detour` within budget,
and (on fixtures with `via`) `coverage` at least as good as Task 10. Record it.

- [ ] **Step 4: Commit**

```bash
git add route_planner.py
git commit -m "feat: tighten road-type and traffic scenic heuristics"
```

---

### Task 12: Refresh CLAUDE.md

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Update the stale sections**

In `CLAUDE.md`, replace the "Known State of the Code" section (and any references to
`scenic_factorify`, `yen_k_shortest_routes`, and "shortest path weighted by travel
time" as the whole story) with an accurate description:

```markdown
## Architecture (current)

- `route_planner.py` — graph construction + `score_scenic_edges()` (relative-proximity
  scenic scoring: curviness, road type, speed, traffic, and percentile-ranked water /
  nature / viewpoint proximity) + `plan_scenic_route()` (fast route by travel_time,
  scenic route by `scenic_cost`, capped at `max_detour`) + `route_coords()`.
- `app.py` — Flask routes, geocoding, SSE progress stream (`/stream`), Folium map,
  TomTom real-time closure avoidance, Google Maps links.
- `evaluate.py` + `evaluation/` — validation harness: fixtures (start/end/via) scored
  for waypoint coverage, fast-vs-scenic divergence, detour ratio, and scenic gain.
- `tests/` — pytest unit tests for the metric and scoring helpers.

## Validating scenic-route quality

Run `python evaluate.py` to print the scorecard. Add trips to `evaluation/fixtures.py`
with `via` place/road names describing the intended scenic route. High `ndiv`/`ldiv`
means the scenic route genuinely differs from the fastest one; `coverage` measures how
well it matches your reference detour; `detour` must stay within each fixture's cap.

## Known state

- `database_speeds.py` has a hardcoded DB password and is not wired in (unused).
- Graph and geocode caches are on disk; delete `.graph_cache/` to force re-fetch.
- Speed/performance of graph loading is a known, separate concern (not yet addressed).
```

- [ ] **Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: refresh CLAUDE.md to match current architecture"
```

---

## Final verification

- [ ] Run the full unit suite: `python -m pytest -q` — Expected: all tests PASS.
- [ ] Run the harness: `python evaluate.py` — Expected: scenic routes diverge from
  fast routes (`ndiv`/`ldiv` clearly above the Task 8 baseline), detour ratios within
  each fixture's cap, and `coverage` reported for fixtures with `via` waypoints.
- [ ] Confirm the app still runs: `python app.py`, open http://localhost:5001, plan
  the default route — the scenic (green) line should now visibly differ from the fast
  (blue) line.

## Self-Review notes (author)

- **Spec coverage:** harness fixtures/metrics/runner (Tasks 2-8), relative-proximity
  fix (Tasks 9-10), road-type/traffic tightening (Task 11), shared `route_coords`
  (Task 6), CLAUDE.md refresh (Task 12), unit tests for metrics (Tasks 2-5) and the
  proximity helper (Task 9). Speed explicitly out of scope per spec.
- **Reference-free metrics** (divergence, detour, gain) let the seed fixture prove the
  fix even before the user adds `via`-rich fixtures.
- **Validation honesty:** scoring-quality tasks (10, 11) are verified by before/after
  scorecards, not by asserting a hard-coded pass — the plan says to reject changes that
  don't improve divergence.
- **Review amendments (2026-07-03):** (1) viewpoints keep an absolute 200 m gradient —
  percentile ranking would zero out sparse features (7 viewpoints in the Ithaca bbox
  puts the median edge past the far gate); relative scoring applies only to dense
  water/nature. (2) fixtures docstring now warns against bare long-road `via` names,
  which Nominatim geocodes to one arbitrary point and cause false coverage failures.
