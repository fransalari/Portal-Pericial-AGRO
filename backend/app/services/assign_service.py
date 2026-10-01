from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from app.services.geo_utils import points_to_feature_collection, project_geom


def assign_nearest_damage(
    centroid_lons: np.ndarray,
    centroid_lats: np.ndarray,
    sample_lons: np.ndarray,
    sample_lats: np.ndarray,
    sample_damage: np.ndarray,
    geom_wgs84=None,
) -> dict[str, Any]:
    """Assign each centroid the damage of the spatially nearest real sample."""
    if len(sample_lons) == 0:
        raise ValueError("No hay muestras reales para asignar daño a los centroides.")

    if geom_wgs84 is not None:
        _p, forward, _i, _e = project_geom(geom_wgs84)
        c_xy = np.column_stack(forward.transform(centroid_lons, centroid_lats))
        s_xy = np.column_stack(forward.transform(sample_lons, sample_lats))
    else:
        lat0 = float(np.mean(centroid_lats))
        m_lat = 111_320.0
        m_lon = 111_320.0 * np.cos(np.radians(lat0))
        c_xy = np.column_stack((centroid_lons * m_lon, centroid_lats * m_lat))
        s_xy = np.column_stack((sample_lons * m_lon, sample_lats * m_lat))

    tree = cKDTree(s_xy)
    distances, indices = tree.query(c_xy, k=1)
    assigned = np.asarray(sample_damage[indices], dtype=float)

    props = [
        {
            "id": int(i),
            "cluster": int(i),
            "damage": float(assigned[i]),
            "nearest_sample_idx": int(indices[i]),
            "distance_m": float(distances[i]),
        }
        for i in range(len(centroid_lons))
    ]
    return {
        "damages": assigned,
        "distances_m": np.asarray(distances, dtype=float),
        "nearest_indices": np.asarray(indices, dtype=int),
        "centroids_geojson": points_to_feature_collection(centroid_lons, centroid_lats, props),
        "damage_min": float(np.nanmin(assigned)),
        "damage_max": float(np.nanmax(assigned)),
        "damage_mean": float(np.nanmean(assigned)),
    }
