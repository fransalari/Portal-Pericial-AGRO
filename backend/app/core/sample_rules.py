"""Minimum sample count based on lot area (hectares)."""

from __future__ import annotations

import math


def min_samples_for_area_ha(area_ha: float) -> int:
    """Return the minimum number of sampling points for a given area in hectares."""
    if area_ha is None or area_ha <= 0:
        return 5
    a = float(area_ha)
    if a <= 50:
        return 5
    if a <= 100:
        return 6
    if a <= 150:
        return 8
    if a <= 200:
        return 10
    if a <= 300:
        return 12
    if a <= 400:
        return 14
    if a <= 500:
        return 16
    if a <= 700:
        return 18
    if a <= 1000:
        return 20
    # More than 1000 ha: 20 + 1 additional sample every 100 ha
    extra = math.ceil((a - 1000) / 100)
    return 20 + int(extra)
