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


def waypoint_coverage(via, polyline):
    """Fraction of via points the polyline passes within their thresholds.

    via is a list of ((lat, lon), threshold_m) pairs — road-name references get
    looser thresholds than towns/landmarks because Nominatim resolves a road to
    one arbitrary point on it. Returns None when there are no via points.
    """
    if not via:
        return None
    hits = sum(1 for point, threshold_m in via
               if point_to_polyline_m(point, polyline) <= threshold_m)
    return hits / len(via)


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
    """Travel-time-weighted mean edge scenic_score along a path.

    Time weighting reflects what the driver experiences: a scenic edge you spend
    five minutes on counts more than a 50 m connector.
    """
    total_time = 0.0
    weighted = 0.0
    for _, _, data in _path_edges(G, path):
        tt = data.get('travel_time', 0.0)
        weighted += data.get('scenic_score', 0.0) * tt
        total_time += tt
    return weighted / total_time if total_time else 0.0
