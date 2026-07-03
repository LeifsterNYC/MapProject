"""Test trips for the scenic-route validation harness.

Each fixture:
  name       - human label for the scorecard
  start/end  - addresses (geocoded via Nominatim)
  via        - (place-name, threshold_m) pairs the scenic route SHOULD pass near.
               Towns/parks/landmarks geocode to tight points; road names resolve
               to ONE arbitrary point on the road, so they need looser thresholds.
  max_detour - allowed scenic_time / fast_time ratio for this trip
"""

FIXTURES = [
    {
        # Known-good scenic detour: Sand Bank Rd -> West King Rd -> Danby Rd
        # (fun, fast, curvy) instead of straight through town.
        "name": "TJ's Ithaca -> Dryden Rd (Sand Bank)",
        "start": "Trader Joe's, Ithaca, NY",
        "end": "205 Dryden Rd, Ithaca, NY 14850",
        "via": [("Sand Bank Road, Ithaca, NY", 1200),
                ("West King Road, Ithaca, NY", 1200)],
        "max_detour": 3.0,
    },
    {
        # Rt 44/55 through the Gunks (viewpoints) instead of over Mohonk.
        "name": "Accord -> New Paltz (Gunks 44/55)",
        "start": "351 Cooper St, Accord, NY 12404",
        "end": "New Paltz, NY",
        # NB: name-based via points failed twice here: "Minnewaska State Park
        # Preserve" geocodes to the roadless park interior, and "Minnewaska
        # Trail" matches a residential road in Kerhonkson town. Use an explicit
        # point on the US-44/NY-55 ridge (near the Minnewaska viewpoints), and
        # skip Kerhonkson town — the natural scenic approach passes east of it.
        "via": [("US 44/NY 55 Gunks ridge", 1500, (41.735, -74.19))],
        "max_detour": 3.0,
    },
    {
        # Rt 97 along the Delaware (Hawk's Nest) instead of NY-17/I-84.
        "name": "Hancock -> Port Jervis (Rt 97)",
        "start": "Hancock, NY",
        "end": "Port Jervis, NY",
        "via": [("Narrowsburg, NY", 1500),
                ("Barryville, NY", 1500)],
        "max_detour": 3.0,
    },
    {
        # Smoke case: the fast route (Rt 89 lakeside) already IS the scenic one.
        # Expected: little divergence and NO wandering through city streets.
        "name": "Ithaca -> Taughannock (smoke)",
        "start": "312 Thurston Ave, Ithaca, NY 14850",
        "end": "1781 Taughannock Blvd, Ulysses, NY 14886",
        "via": [("Taughannock Falls State Park, NY", 1500)],
        "max_detour": 3.0,
    },
    {
        # NORWAY (international check): Fv 815 along the Vestfjorden shore —
        # curvier and more scenic than the inland E10, barely longer.
        # Exercises fjord-coastline water, numeric-ref rescue, km/h parsing.
        # NB: Stamsund village is on a dead-end Fv 817 spur 4.1 km off the
        # 815 — reference the road itself (same lesson as Minnewaska).
        "name": "Henningsvaer -> Ballstad (Fv 815)",
        "start": "Henningsvær, Norway",
        "end": "Ballstad, Norway",
        "via": [("Fv 815 Vestfjorden corridor", 2000, (68.156, 13.780))],
        "max_detour": 3.0,
    },
    {
        # LONG TRIP (highway-class graph): NY-17 through the Catskills, exit at
        # Hancock onto Rt 97 along the Delaware to Port Jervis, then back east
        # to I-87/Palisades — instead of the I-81/380/80 slog. Generous
        # thresholds: corridor-level routing.
        "name": "Ithaca -> NYC (17/97 long)",
        "start": "312 Thurston Ave, Ithaca, NY 14850",
        "end": "Columbus Circle, New York, NY",
        "via": [("Hancock, NY", 3000),
                ("Barryville, NY", 3000),
                ("Port Jervis, NY", 3000)],
        "max_detour": 3.0,
    },
]
