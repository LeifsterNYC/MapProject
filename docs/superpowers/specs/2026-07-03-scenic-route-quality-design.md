# Scenic Route Quality — Design

**Date:** 2026-07-03
**Status:** Approved; amended after pre-implementation prototype review (see Amendments)
**App:** Ramble (Flask scenic route planner)

## Problem

Ramble computes a "scenic" driving route as an alternative to the fastest route,
but the scenic route is **not meaningfully different from the fast route**, and its
choices feel wrong / not scenic enough.

### Root cause (diagnosed)

The per-edge scenic scoring **does not discriminate**. On the default Ithaca → Taughannock
route:

- The graph carries ~8,000 water points and ~9,900 nature points. With absolute match
  radii (400 m water, 150 m nature), in a green region *almost every edge* qualifies as
  "near water" and "near nature." Those two signals — the heart of "scenic" — are
  effectively **constant (~1.0) across the whole graph** and differentiate nothing.
- `scenic_score` clusters tightly (min 0.34, median 1.64, max 4.25 vs. a theoretical max
  of 6.0). Nearly every edge looks "medium scenic."
- Consequence: even with the **detour cap removed**, the scenic shortest-path shares
  **94 of 110 nodes** with the fastest route. The optimizer has almost no signal to pull
  it onto a prettier path.

Secondary contributor to "feels wrong": the road-type/traffic heuristics rank
`unclassified` = 1.0 and default (no-lane) roads highly, which can steer routes onto
low-quality tracks.

Speed (~21 s dominated by graph load) is a **separate, out-of-scope** issue for this work.

## Goal

Make the scenic route genuinely scenic **and make "is it right?" measurable**, so scoring
changes are validated against ground truth instead of tuned blind.

## Non-goals (YAGNI)

- Speed / caching improvements (separate follow-up)
- UI changes, TomTom, PostgreSQL speed data
- Learned/ML weighting — heuristic tuning against the harness is sufficient for v1

## Approach

Two parts: build a validation harness (the ruler), then fix the scoring signal and measure
every change against it.

### Part 1 — Validation harness (the oracle)

**Fixtures.** A small set (target 5–8) of test trips. Each fixture:

```python
{
  "name": "Ithaca -> Taughannock (lakeside)",
  "start": "312 Thurston Ave, Ithaca, NY 14850",
  "end":   "1781 Taughannock Blvd, Ulysses, NY 14886",
  "via":   ["Trumansburg, NY", "Taughannock Falls State Park"],  # place/road names
  "max_detour": 3.0,
}
```

`via` is a list of **place/road-name strings** the scenic route should pass near,
captured from the user's knowledge or an online scenic detour. The harness geocodes each
name to a point (reusing the existing Nominatim geocoder). This is the chosen capture
format: lowest friction, no coordinate entry.

Fixtures live in `evaluation/fixtures.py` (plain Python list of dicts) — easy to edit,
no parsing layer.

**Metrics.** For each fixture the harness computes:

1. **Waypoint coverage** (needs `via`): fraction of `via` points the computed scenic
   polyline passes within a threshold (~300 m) of. Primary correctness signal.
2. **Divergence** (reference-free): how much the scenic route differs from the fast route,
   measured by shared-node fraction and shared-length fraction. This is the headline
   number today (~85% shared → near-identical).
3. **Detour ratio** (reference-free): scenic travel time ÷ fast travel time; must stay
   ≤ `max_detour`.
4. **Scenic gain** (reference-free): mean edge `scenic_score` along the scenic path vs.
   along the fast path; should be clearly higher.

**Runner.** `evaluate.py` executes every fixture through the existing routing pipeline and
prints a per-fixture + aggregate scorecard (plain text table). This is the before/after
ruler for all scoring changes. It reuses `initialize_graph`, `score_scenic_edges`,
`plan_scenic_route`, and the `_route_coords` geometry helper (which may be lifted from
`app.py` into a shared module so the harness and Flask app share one implementation).

Geocoding and graph fetches are cached (existing on-disk graph cache + a small geocode
cache for `via` names) so repeated harness runs are fast and don't hammer Nominatim.

### Part 2 — Fix the scenic score (measured against Part 1)

Rework `score_scenic_edges` in `route_planner.py` so components discriminate:

- **Relative proximity, not absolute.** Convert water/nature proximity to a
  **percentile/relative** signal within the current graph: an edge earns water/nature
  credit only if it is closer to water/nature than typical for *this* route's bbox. In a
  green region this separates genuine riverside/forest roads from ordinary ones instead of
  saturating every edge near 1.0.
- **Rebalance components** so no one or two saturating signals dominate; ensure curviness,
  quiet-road, and speed-sweet-spot signals meaningfully contribute.
- **Scenery-per-minute objective.** Shape the scenic cost so a short detour to something
  genuinely scenic wins, while a long detour through mediocre roads does not. (Refine the
  existing `travel_time * exp(k * (1 - t))` shaping and/or the detour handling.)
- **Tighten road-type/traffic heuristics** so `unclassified`/`service` tracks are not
  over-rewarded relative to pleasant `tertiary`/`secondary` country roads.

Each change is accepted only if the scorecard improves: divergence and waypoint coverage
rise while detour ratio stays within budget across the fixture set. No change is judged by
eyeballing a single map.

## Data flow

```
fixtures.py ── (start,end,via,max_detour) ──> evaluate.py
   evaluate.py:
     geocode start/end/via  ──> initialize_graph(bbox)  [cached]
     score_scenic_edges(G, weights)                     [Part 2 under test]
     plan_scenic_route(G, s, e, max_detour) -> fast, scenic, times
     _route_coords(G, scenic) -> polyline
     metrics: coverage(via, polyline), divergence(fast,scenic),
              detour_ratio(times), scenic_gain(G,fast,scenic)
   -> scorecard (per-fixture + aggregate)
```

The Flask app is unchanged in behavior; it benefits automatically from the improved
`score_scenic_edges`.

## Error handling

- A fixture whose start/end/via fails to geocode is reported as an error row in the
  scorecard (does not abort the whole run).
- If `plan_scenic_route` returns the fast route as scenic (no valid detour within cap),
  that fixture shows divergence = 0 and coverage as computed — this is a *result*, not a
  crash, and is exactly the failure signal we want visible.
- Network calls (Nominatim) are wrapped with the existing timeout/exception handling;
  cached results avoid repeated calls.

## Testing

- The harness itself is the primary test artifact for scenic quality.
- Add focused unit tests for the new metric functions (coverage, divergence, detour ratio,
  scenic gain) using tiny synthetic graphs/polylines with known answers — these must not
  depend on network access.
- The scoring rework is validated by the scorecard on real fixtures (network + cached
  graphs), run manually.

## Structure / files touched

- `evaluation/fixtures.py` — **new** — test trips (start/end/via/max_detour)
- `evaluate.py` — **new** — runner + scorecard
- `evaluation/metrics.py` — **new** — coverage / divergence / detour / scenic-gain (unit-tested)
- `route_planner.py` — **modified** — `score_scenic_edges` rework
- shared route-geometry helper — `_route_coords` possibly lifted from `app.py` into a
  shared module so app and harness agree
- `CLAUDE.md` — **updated** — it is stale (describes stubs that are long implemented)

## Success criteria

1. `evaluate.py` runs the fixture set and prints a scorecard.
2. Waypoint coverage is high on the fixtures where the user encoded a known scenic detour.
3. Scenic gain per extra minute is clearly positive on fixtures where a genuinely
   better alternative exists, and the smoke fixture does NOT wander (see Amendment B).
4. Detour ratios stay within each fixture's `max_detour`.
5. The improvement is demonstrated by before/after scorecards, not by a single map.

## Amendments (2026-07-03, post-review — evidence-based)

A prototype of the original Part 2 was run against the cached Ithaca graph before
implementation. Findings and resulting amendments:

**A. Fixtures are a hard prerequisite, and the seed fixture flips meaning.**
For Ithaca → 1781 Taughannock Blvd the fastest route already IS the scenic one
(Rt 89 along the lake, past Taughannock park). No divergence is correct there; it
becomes a *don't-wander* smoke case. Real fixtures with known scenic detours
(user-supplied): Trader Joe's Ithaca → 205 Dryden Rd via Sand Bank Rd/West King
Rd/Danby Rd; 351 Cooper St Accord → New Paltz via Rt 44/55 through the Gunks
(Kerhonkson, Minnewaska) rather than over Mohonk; Hancock → Port Jervis via Rt 97
along the Delaware (Narrowsburg, Barryville) rather than 17/I-84. Ithaca → NYC via
17 + Palisades is recorded as future long-trip work (exceeds the 50-mile app cap).

**B. Divergence is demoted from headline metric to diagnostic.** The prototype
produced routes with node-divergence 0.41–0.55 that merely zigzagged through
residential Ithaca streets with negligible scenic gain (2.49 → 2.54). Divergence is
gameable. Headline metrics: **waypoint coverage** and **scenic gain per extra
minute** (travel-time-weighted mean scenic score, scenic vs fast, per extra minute
spent). `ndiv`/`ldiv` remain on the scorecard as diagnostics.

**C. Normalization must be empirical, not theoretical-max.** With the planned
divide-by-max-score normalization, summed edge scores cluster at 1.5–2.3 of 6.0;
the normalized signal spans ~0.25–0.39 and the exp shaping differentiates nothing
(scenic ≈ fast at every K tested). Fix: percentile-rank the *total* edge score
across the graph → `scenic_t` ∈ [0, 1], uniformly spread by construction.

**D. Fixed `_SCENIC_K` + cliff fallback is replaced by an adaptive detour search.**
Behavior is knife-edged in K (K=3: identical to fast; K=4: 97%-different route at
1.9x time on the test trip). `plan_scenic_route` now probes a ladder of K values
(one Dijkstra each — cheap), keeps candidates within `max_detour`, and returns the
one with the best scenic gain per extra minute. The "no valid detour → silently
return fast" cliff disappears; the fast route remains the graceful floor.

**E. Anti-wandering gate.** Water/nature/curviness bonuses are damped on
`residential` and `service` edges so downtown grids stop attracting the scenic
route (root cause of finding B).
