from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

from app.services.geo_utils import project_geom

DAMAGE_CMAP = [
    (0.0, (34, 139, 34)),      # green
    (25.0, (154, 205, 50)),
    (50.0, (255, 215, 0)),
    (75.0, (255, 140, 0)),
    (100.0, (178, 34, 34)),    # red
]


def _colorize(values: np.ndarray, vmin: float, vmax: float, discrete: bool) -> np.ndarray:
    """Map values to RGBA uint8 array."""
    out = np.zeros((*values.shape, 4), dtype=np.uint8)
    valid = np.isfinite(values)
    if not valid.any():
        return out

    if vmax <= vmin:
        vmax = vmin + 1.0

    if discrete:
        # 5 classes
        edges = np.linspace(vmin, vmax, 6)
        colors = [
            (34, 139, 34, 200),
            (154, 205, 50, 200),
            (255, 215, 0, 200),
            (255, 140, 0, 200),
            (178, 34, 34, 200),
        ]
        classes = np.digitize(values, edges[1:-1], right=True)
        for i, rgba in enumerate(colors):
            mask = valid & (classes == i)
            out[mask] = rgba
    else:
        t = np.clip((values - vmin) / (vmax - vmin), 0, 1)
        # interpolate across DAMAGE_CMAP stops in 0-100 space remapped
        stops = np.array([s[0] for s in DAMAGE_CMAP]) / 100.0
        cols = np.array([s[1] for s in DAMAGE_CMAP], dtype=float)
        flat = t[valid]
        r = np.interp(flat, stops, cols[:, 0])
        g = np.interp(flat, stops, cols[:, 1])
        b = np.interp(flat, stops, cols[:, 2])
        out[valid, 0] = r.astype(np.uint8)
        out[valid, 1] = g.astype(np.uint8)
        out[valid, 2] = b.astype(np.uint8)
        out[valid, 3] = 200

    out[~valid, 3] = 0
    return out


def idw_interpolate(
    geom_wgs84,
    point_lons: np.ndarray,
    point_lats: np.ndarray,
    values: np.ndarray,
    res_m: float,
    power: float = 2.0,
    scale_mode: Literal["fixed", "adaptive"] = "fixed",
    symbology: Literal["continuous", "discrete"] = "continuous",
    out_png: Path | None = None,
    out_npy: Path | None = None,
) -> dict[str, Any]:
    """IDW over a regular grid clipped to the lot polygon. Writes PNG overlay + npy."""
    projected, forward, inverse, _epsg = project_geom(geom_wgs84)
    minx, miny, maxx, maxy = projected.bounds
    xs = np.arange(minx + res_m / 2, maxx, res_m)
    ys = np.arange(miny + res_m / 2, maxy, res_m)
    if len(xs) == 0 or len(ys) == 0:
        xs = np.array([(minx + maxx) / 2])
        ys = np.array([(miny + maxy) / 2])

    xx, yy = np.meshgrid(xs, ys)
    # note: row 0 is south (miny); for image we flip later
    grid_x = xx.ravel()
    grid_y = yy.ravel()

    from shapely.geometry import Point
    from shapely import prepared

    prep = prepared.prep(projected)
    mask = np.fromiter(
        (prep.contains(Point(x, y)) for x, y in zip(grid_x, grid_y)),
        dtype=bool,
        count=len(grid_x),
    )

    src_xy = np.column_stack(forward.transform(point_lons, point_lats))
    vals = np.asarray(values, dtype=float)
    tree = cKDTree(src_xy)

    raster = np.full(grid_x.shape, np.nan, dtype=float)
    if mask.any():
        dists, idxs = tree.query(np.column_stack((grid_x[mask], grid_y[mask])), k=min(8, len(vals)))
        if dists.ndim == 1:
            dists = dists[:, None]
            idxs = idxs[:, None]
        # avoid div by zero
        zero = dists[:, 0] < 1e-6
        weights = 1.0 / np.power(np.maximum(dists, 1e-6), power)
        weights_sum = weights.sum(axis=1)
        interp = (weights * vals[idxs]).sum(axis=1) / weights_sum
        interp[zero] = vals[idxs[zero, 0]]
        raster[mask] = interp

    raster2d = raster.reshape((len(ys), len(xs)))

    finite = raster2d[np.isfinite(raster2d)]
    if scale_mode == "adaptive" and len(finite) > 0:
        vmin = float(np.percentile(finite, 5))
        vmax = float(np.percentile(finite, 95))
        if vmax <= vmin:
            vmin, vmax = float(np.min(finite)), float(np.max(finite))
    else:
        vmin, vmax = 0.0, 100.0

    rgba = _colorize(raster2d, vmin, vmax, discrete=(symbology == "discrete"))
    # flip vertically so north is up for map overlay
    rgba_img = np.flipud(rgba)
    raster_north = np.flipud(raster2d)

    # geographic bounds for image overlay (west, south, east, north) in WGS84
    corners_m = [(minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy)]
    corners_ll = [inverse.transform(x, y) for x, y in corners_m]
    west = min(c[0] for c in corners_ll)
    east = max(c[0] for c in corners_ll)
    south = min(c[1] for c in corners_ll)
    north = max(c[1] for c in corners_ll)

    if out_png:
        out_png.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rgba_img, mode="RGBA").save(out_png)
    if out_npy:
        out_npy.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_npy, raster_north)

    mean_damage = float(np.nanmean(finite)) if len(finite) else float("nan")
    return {
        "width": int(len(xs)),
        "height": int(len(ys)),
        "bounds": [west, south, east, north],
        "vmin": vmin,
        "vmax": vmax,
        "scale_mode": scale_mode,
        "symbology": symbology,
        "power": power,
        "mean_damage": mean_damage,
        "png_path": str(out_png) if out_png else None,
        "npy_path": str(out_npy) if out_npy else None,
        "raster": raster_north,
    }


ZONE_LABELS = ["Muy bajo", "Bajo", "Medio", "Alto", "Muy alto"]


def zonify_raster(
    raster: np.ndarray,
    scale_mode: Literal["fixed", "adaptive"] = "fixed",
    symbology: Literal["continuous", "discrete"] = "discrete",
    out_png: Path | None = None,
    bounds: list[float] | None = None,
) -> dict[str, Any]:
    finite = raster[np.isfinite(raster)]
    if scale_mode == "adaptive" and len(finite) > 0:
        vmin = float(np.percentile(finite, 5))
        vmax = float(np.percentile(finite, 95))
        if vmax <= vmin:
            vmin, vmax = float(np.min(finite)), float(np.max(finite))
    else:
        vmin, vmax = 0.0, 100.0

    # Always classify into 5 damage zones for stats; visualization may be continuous
    edges = np.linspace(vmin, vmax, 6)
    classes = np.full(raster.shape, -1, dtype=int)
    valid = np.isfinite(raster)
    classes[valid] = np.digitize(raster[valid], edges[1:-1], right=True)

    rgba = _colorize(raster, vmin, vmax, discrete=(symbology == "discrete"))
    if out_png:
        out_png.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rgba, mode="RGBA").save(out_png)

    zone_stats = []
    total = int(valid.sum()) or 1
    for i, label in enumerate(ZONE_LABELS):
        count = int((classes == i).sum())
        zone_stats.append(
            {
                "zone": i,
                "label": label,
                "count": count,
                "pct_area": round(100.0 * count / total, 2),
                "min": float(edges[i]),
                "max": float(edges[i + 1]),
            }
        )

    return {
        "vmin": vmin,
        "vmax": vmax,
        "edges": edges.tolist(),
        "zone_stats": zone_stats,
        "labels": ZONE_LABELS,
        "scale_mode": scale_mode,
        "symbology": symbology,
        "png_path": str(out_png) if out_png else None,
        "bounds": bounds,
    }
