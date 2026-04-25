"""Geographic clustering tools for the Clustering agent.

Uses a pure-numpy K-means so no heavy ML dependencies are needed.
"""
from __future__ import annotations

from typing import Any

import numpy as np


def cluster_places(
    places: list[dict[str, Any]],
    num_clusters: int,
) -> list[list[dict[str, Any]]]:
    """Partition *places* into *num_clusters* geographically coherent groups.

    Algorithm: K-means on (latitude, longitude) with 100 iterations.
    Falls back gracefully when there are fewer places than clusters.
    """
    if not places:
        return []

    n = len(places)
    try:
        requested = int(num_clusters)
    except (TypeError, ValueError):
        requested = 1
    k = max(1, min(requested, n))

    if k == 1:
        return [list(places)]

    coords = _extract_coords(places)
    if coords is None:
        return _round_robin_clusters(places, k)

    # If coordinates are degenerate (e.g., all points identical), geo clustering
    # cannot separate places meaningfully. Spread items round-robin across days.
    unique_coord_count = len(np.unique(coords, axis=0))
    if unique_coord_count < k:
        return _round_robin_clusters(places, k)

    centers = _init_kmeans_pp(coords, k)

    try:
        for _ in range(100):
            # Assignment step
            dist_matrix = np.stack(
                [_haversine_all(coords, c) for c in centers], axis=1
            )
            assignments = np.argmin(dist_matrix, axis=1)

            # Update step
            new_centers = np.zeros_like(centers)
            for ci in range(k):
                members = coords[assignments == ci]
                new_centers[ci] = members.mean(axis=0) if len(members) else centers[ci]

            if np.allclose(centers, new_centers, atol=1e-8):
                break
            centers = new_centers

        clusters: list[list[dict[str, Any]]] = [[] for _ in range(k)]
        for i, place in enumerate(places):
            clusters[assignments[i]].append(place)

        result = [c for c in clusters if c]
        if result:
            return result
    except Exception:
        pass

    # Fallback: round-robin when k-means fails (e.g. degenerate coordinates)
    return _round_robin_clusters(places, k)


def _round_robin_clusters(
    places: list[dict[str, Any]],
    k: int,
) -> list[list[dict[str, Any]]]:
    k = max(1, int(k))
    buckets: list[list[dict[str, Any]]] = [[] for _ in range(k)]
    for i, place in enumerate(places):
        buckets[i % k].append(place)
    return [b for b in buckets if b]


def _extract_coords(places: list[dict[str, Any]]) -> np.ndarray | None:
    rows: list[list[float]] = []
    for place in places:
        location = place.get("location") if isinstance(place, dict) else None
        if not isinstance(location, dict):
            return None

        try:
            lat = float(location.get("latitude"))
            lng = float(location.get("longitude"))
        except (TypeError, ValueError):
            return None

        if not np.isfinite(lat) or not np.isfinite(lng):
            return None

        rows.append([lat, lng])

    if not rows:
        return None
    return np.array(rows, dtype=float)


def _init_kmeans_pp(coords: np.ndarray, k: int) -> np.ndarray:
    """K-means++ initialisation with safeguards for zero-distance datasets."""
    n = len(coords)
    rng = np.random.default_rng(seed=42)
    center_idx = [int(rng.integers(n))]

    for _ in range(k - 1):
        dists = np.min([_haversine_all(coords, coords[ci]) for ci in center_idx], axis=0)
        total = float(np.sum(dists))

        if not np.isfinite(total) or total <= 0:
            remaining = [i for i in range(n) if i not in center_idx]
            center_idx.append(int(rng.choice(remaining)))
            continue

        probs = dists / total
        if not np.all(np.isfinite(probs)):
            remaining = [i for i in range(n) if i not in center_idx]
            center_idx.append(int(rng.choice(remaining)))
            continue

        center_idx.append(int(rng.choice(n, p=probs)))

    return coords[center_idx]


def _haversine_all(coords: np.ndarray, center: np.ndarray) -> np.ndarray:
    """Vectorised haversine distance (km) from every row in *coords* to *center*."""
    R = 6371.0
    lat1, lng1 = np.radians(coords[:, 0]), np.radians(coords[:, 1])
    lat2, lng2 = np.radians(center[0]), np.radians(center[1])
    dlat = lat2 - lat1
    dlng = lng2 - lng1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlng / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def num_days_from_dates(trip_start: str, trip_end: str) -> int:
    """Compute how many trip days fit between two ISO datetime strings."""
    from datetime import datetime
    import math

    fmt_variants = [
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%d",
    ]
    start = end = None
    for fmt in fmt_variants:
        try:
            start = datetime.strptime(trip_start[:19], fmt[:19])
            end = datetime.strptime(trip_end[:19], fmt[:19])
            break
        except ValueError:
            continue
    if start is None or end is None:
        raise ValueError(f"Cannot parse dates: {trip_start!r}, {trip_end!r}")

    delta_hours = (end - start).total_seconds() / 3600
    return max(1, math.ceil(delta_hours / 24))


# ── LLM tool schema ───────────────────────────────────────────────────────────

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "cluster_places",
            "description": (
                "Group a list of places into geographically coherent clusters "
                "so that each cluster can form one day of sightseeing. "
                "Uses K-means clustering on latitude/longitude."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "places": {
                        "type": "array",
                        "description": "List of place objects with location.latitude and location.longitude.",
                        "items": {"type": "object"},
                    },
                    "num_clusters": {
                        "type": "integer",
                        "description": "Desired number of clusters (= number of trip days).",
                    },
                },
                "required": ["places", "num_clusters"],
            },
        },
    }
]
