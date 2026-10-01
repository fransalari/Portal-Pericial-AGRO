from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.cluster import KMeans

from app.services.geo_utils import cluster_hulls, points_to_feature_collection, project_geom


def run_kmeans(
    lons: np.ndarray,
    lats: np.ndarray,
    k: int,
    geom_wgs84=None,
) -> dict[str, Any]:
    """Cluster grid points with K-Means in projected metres; return centroids + labels."""
    k = max(1, min(int(k), len(lons)))
    if geom_wgs84 is not None:
        _proj, forward, _inv, _epsg = project_geom(geom_wgs84)
        xy = np.column_stack(forward.transform(lons, lats))
    else:
        # approximate degrees -> metres near centroid
        lat0 = float(np.mean(lats))
        m_per_deg_lat = 111_320.0
        m_per_deg_lon = 111_320.0 * np.cos(np.radians(lat0))
        xy = np.column_stack(((lons - np.mean(lons)) * m_per_deg_lon, (lats - np.mean(lats)) * m_per_deg_lat))

    model = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = model.fit_predict(xy)
    centers_xy = model.cluster_centers_

    if geom_wgs84 is not None:
        _proj, forward, inverse, _epsg = project_geom(geom_wgs84)
        c_lons, c_lats = inverse.transform(centers_xy[:, 0], centers_xy[:, 1])
    else:
        lat0 = float(np.mean(lats))
        lon0 = float(np.mean(lons))
        m_per_deg_lat = 111_320.0
        m_per_deg_lon = 111_320.0 * np.cos(np.radians(lat0))
        c_lons = centers_xy[:, 0] / m_per_deg_lon + lon0
        c_lats = centers_xy[:, 1] / m_per_deg_lat + lat0

    c_lons = np.asarray(c_lons, dtype=float)
    c_lats = np.asarray(c_lats, dtype=float)
    labels = np.asarray(labels, dtype=int)

    centroid_props = [{"id": int(i), "cluster": int(i), "label": f"C{i+1}"} for i in range(k)]
    grid_props = [{"id": int(i), "cluster": int(labels[i])} for i in range(len(lons))]

    return {
        "k": k,
        "labels": labels,
        "centroid_lons": c_lons,
        "centroid_lats": c_lats,
        "centroids_geojson": points_to_feature_collection(c_lons, c_lats, centroid_props),
        "grid_labeled_geojson": points_to_feature_collection(lons, lats, grid_props),
        "clusters_geojson": cluster_hulls(lons, lats, labels),
    }
