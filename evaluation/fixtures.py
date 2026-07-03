"""Test trips for the scenic-route validation harness.

Each fixture:
  name       - human label for the scorecard
  start/end  - addresses (geocoded via Nominatim)
  via        - (place-name, threshold_m) pairs the scenic route SHOULD pass near.
               Towns/parks/landmarks geocode to tight points; road names resolve
               to ONE arbitrary point on the road, so they need looser thresholds.
  max_detour - allowed scenic_time / fast_time ratio for this trip

Future (out of harness scale for now): Ithaca -> NYC preferring NY-17 + Palisades
Pkwy over I-380/I-80 — exceeds the 50-mile app cap and needs a huge graph.
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
        "via": [("Kerhonkson, NY", 1200),
                ("Minnewaska State Park Preserve", 2500)],
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
]
