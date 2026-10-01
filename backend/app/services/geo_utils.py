from __future__ import annotations

import math
from typing import Any

import numpy as np
from pyproj import Transformer
from shapely.geometry import MultiPoint, Point, Polygon, mapping, shape
from shapely.ops import transform as shp_transform


def geom_from_geojson(geom: dict[str, Any]):
    return shape(geom)


def area_ha_wgs84(geom) -> float:
    """Geodesic-ish area via local UTM projection."""
    centroid = geom.centroid
    utm_epsg = _utm_epsg(centroid.x, centroid.y)
    projector = Transformer.from_crs("EPSG:4326", f"EPSG:{utm_epsg}", always_xy=True)
    projected = shp_transform(lambda x, y: projector.transform(x, y), geom)
    return float(projected.area / 10_000.0)


def _utm_epsg(lon: float, lat: float) -> int:
    zone = int((lon + 180) // 6) + 1
    return (32600 if lat >= 0 else 32700) + zone


def project_geom(geom, lon: float | None = None, lat: float | None = None):
    c = geom.centroid
    lon = lon if lon is not None else c.x
    lat = lat if lat is not None else c.y
    epsg = _utm_epsg(lon, lat)
    forward = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    inverse = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)
    projected = shp_transform(lambda x, y: forward.transform(x, y), geom)
    return projected, forward, inverse, epsg


def grid_points_in_polygon(geom_wgs84, res_m: float) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """
    Generate interior sample points on a regular grid at resolution res_m (metres).
    Returns lon/lat arrays and metadata.
    """
    projected, _fwd, inverse, epsg = project_geom(geom_wgs84)
    minx, miny, maxx, maxy = projected.bounds
    xs = np.arange(minx + res_m / 2, maxx, res_m)
    ys = np.arange(miny + res_m / 2, maxy, res_m)
    if len(xs) == 0 or len(ys) == 0:
        # fallback: centroid only
        c = geom_wgs84.centroid
        return np.array([c.x]), np.array([c.y]), {
            "n_cells": 1,
            "grid_res_m": res_m,
            "epsg": epsg,
            "width": 1,
            "height": 1,
        }

    xx, yy = np.meshgrid(xs, ys)
    flat_x = xx.ravel()
    flat_y = yy.ravel()
    mask = np.fromiter(
        (projected.contains(Point(x, y)) or projected.touches(Point(x, y)) for x, y in zip(flat_x, flat_y)),
        dtype=bool,
        count=len(flat_x),
    )
    # Prefer strict interior; if empty use prepared contains with buffer tolerance
    if not mask.any():
        buffered = projected.buffer(res_m * 0.1)
        mask = np.fromiter(
            (buffered.contains(Point(x, y)) for x, y in zip(flat_x, flat_y)),
            dtype=bool,
            count=len(flat_x),
        )

    sel_x = flat_x[mask]
    sel_y = flat_y[mask]
    if len(sel_x) == 0:
        c = projected.centroid
        sel_x = np.array([c.x])
        sel_y = np.array([c.y])

    lons, lats = inverse.transform(sel_x, sel_y)
    lons = np.asarray(lons, dtype=float)
    lats = np.asarray(lats, dtype=float)
    meta = {
        "n_cells": int(len(lons)),
        "grid_res_m": float(res_m),
        "epsg": epsg,
        "width": int(len(xs)),
        "height": int(len(ys)),
        "bbox_m": [float(minx), float(miny), float(maxx), float(maxy)],
    }
    return lons, lats, meta


def grid_outline_geojson(geom_wgs84, res_m: float, max_lines: int = 80) -> dict[str, Any]:
    """Lightweight grid lines for map visualization (not every cell polygon)."""
    projected, _fwd, inverse, _epsg = project_geom(geom_wgs84)
    minx, miny, maxx, maxy = projected.bounds
    width = maxx - minx
    height = maxy - miny
    nx = max(2, min(max_lines, int(math.ceil(width / res_m)) + 1))
    ny = max(2, min(max_lines, int(math.ceil(height / res_m)) + 1))
    step_x = width / (nx - 1) if nx > 1 else res_m
    step_y = height / (ny - 1) if ny > 1 else res_m

    features = []
    for i in range(nx):
        x = minx + i * step_x
        line_m = [(x, miny), (x, maxy)]
        line_ll = [list(inverse.transform(px, py)) for px, py in line_m]
        features.append(
            {
                "type": "Feature",
                "properties": {"orient": "v"},
                "geometry": {"type": "LineString", "coordinates": line_ll},
            }
        )
    for j in range(ny):
        y = miny + j * step_y
        line_m = [(minx, y), (maxx, y)]
        line_ll = [list(inverse.transform(px, py)) for px, py in line_m]
        features.append(
            {
                "type": "Feature",
                "properties": {"orient": "h"},
                "geometry": {"type": "LineString", "coordinates": line_ll},
            }
        )
    return {"type": "FeatureCollection", "features": features}


def points_to_feature_collection(
    lons: np.ndarray,
    lats: np.ndarray,
    properties: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    features = []
    for i, (lon, lat) in enumerate(zip(lons, lats)):
        props = properties[i] if properties else {"id": i}
        features.append(
            {
                "type": "Feature",
                "properties": props,
                "geometry": {"type": "Point", "coordinates": [float(lon), float(lat)]},
            }
        )
    return {"type": "FeatureCollection", "features": features}


def polygon_feature(geom, properties: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "type": "Feature",
        "properties": properties or {},
        "geometry": mapping(geom),
    }


def cluster_hulls(lons: np.ndarray, lats: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    features = []
    for label in sorted(set(labels.tolist())):
        mask = labels == label
        pts = list(zip(lons[mask], lats[mask]))
        if len(pts) < 3:
            geom = Point(pts[0]) if pts else None
            if geom is None:
                continue
            gtype = "Point"
            coords = [pts[0][0], pts[0][1]]
            features.append(
                {
                    "type": "Feature",
                    "properties": {"cluster": int(label)},
                    "geometry": {"type": gtype, "coordinates": coords},
                }
            )
            continue
        hull = MultiPoint(pts).convex_hull
        features.append(
            {
                "type": "Feature",
                "properties": {"cluster": int(label)},
                "geometry": mapping(hull),
            }
        )
    return {"type": "FeatureCollection", "features": features}
