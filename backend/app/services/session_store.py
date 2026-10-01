from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.core.config import settings
from app.core.sample_rules import min_samples_for_area_ha
from app.models.schemas import IdwParams, SessionState, ZoneParams
from app.services.assign_service import assign_nearest_damage
from app.services.catalog import catalog
from app.services.compare_service import compare_methods, summary_stats
from app.services.geo_utils import (
    area_ha_wgs84,
    cluster_hulls,
    geom_from_geojson,
    grid_outline_geojson,
    grid_points_in_polygon,
    points_to_feature_collection,
    polygon_feature,
)
from app.services.idw_service import idw_interpolate, zonify_raster
from app.services.kmeans_service import run_kmeans


class SessionStore:
    def __init__(self) -> None:
        settings.sessions_dir.mkdir(parents=True, exist_ok=True)
        self._mem: dict[str, dict[str, Any]] = {}

    def _dir(self, session_id: str) -> Path:
        d = settings.sessions_dir / session_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _save_meta(self, session_id: str) -> None:
        meta = self._mem[session_id]["state"]
        path = self._dir(session_id) / "state.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(meta.model_dump() if isinstance(meta, SessionState) else meta, f, indent=2)

    def _persist_arrays(self, session_id: str) -> None:
        """Write runtime arrays so sessions survive process restarts."""
        sess = self._mem[session_id]
        sdir = self._dir(session_id)
        if sess.get("grid_lons") is not None and sess.get("grid_lats") is not None:
            np.save(sdir / "grid_lons.npy", sess["grid_lons"])
            np.save(sdir / "grid_lats.npy", sess["grid_lats"])
        if sess.get("labels") is not None:
            np.save(sdir / "labels.npy", sess["labels"])
        if sess.get("centroid_lons") is not None and sess.get("centroid_lats") is not None:
            payload: dict[str, np.ndarray] = {
                "lons": np.asarray(sess["centroid_lons"], dtype=float),
                "lats": np.asarray(sess["centroid_lats"], dtype=float),
            }
            if sess.get("centroid_damages") is not None:
                payload["damages"] = np.asarray(sess["centroid_damages"], dtype=float)
            np.savez(sdir / "centroids.npz", **payload)

    def _coords_from_geojson(self, path: Path) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
        with path.open(encoding="utf-8") as f:
            fc = json.load(f)
        lons: list[float] = []
        lats: list[float] = []
        props: list[dict[str, Any]] = []
        for feat in fc.get("features", []):
            geom = feat.get("geometry") or {}
            if geom.get("type") != "Point":
                continue
            coords = geom.get("coordinates") or []
            if len(coords) < 2:
                continue
            lons.append(float(coords[0]))
            lats.append(float(coords[1]))
            props.append(feat.get("properties") or {})
        return np.asarray(lons, dtype=float), np.asarray(lats, dtype=float), props

    def _hydrate(self, session_id: str) -> None:
        sdir = settings.sessions_dir / session_id
        state_path = sdir / "state.json"
        if not state_path.exists():
            raise KeyError(f"Sesión no encontrada: {session_id}")

        with state_path.open(encoding="utf-8") as f:
            state = SessionState.model_validate(json.load(f))

        lot_path = sdir / "lot.geojson"
        if not lot_path.exists():
            raise KeyError(f"Sesión incompleta (sin lote): {session_id}")
        with lot_path.open(encoding="utf-8") as f:
            lot_fc = json.load(f)
        lot_feat = (lot_fc.get("features") or [None])[0]
        if not lot_feat or not lot_feat.get("geometry"):
            raise KeyError(f"Sesión incompleta (geometría inválida): {session_id}")
        geom = geom_from_geojson(lot_feat["geometry"])

        samples_path = sdir / "samples.parquet"
        if samples_path.exists():
            samples = pd.read_parquet(samples_path)
        else:
            samples = catalog.samples_for_lot(state.lot_id)

        grid_lons = grid_lats = labels = None
        centroid_lons = centroid_lats = centroid_damages = None
        idw_raster = idw_bounds = None

        if (sdir / "grid_lons.npy").exists() and (sdir / "grid_lats.npy").exists():
            grid_lons = np.load(sdir / "grid_lons.npy")
            grid_lats = np.load(sdir / "grid_lats.npy")
        elif (sdir / "grid_labeled.geojson").exists():
            grid_lons, grid_lats, gprops = self._coords_from_geojson(sdir / "grid_labeled.geojson")
            if gprops and "cluster" in gprops[0]:
                labels = np.asarray([int(p.get("cluster", 0)) for p in gprops], dtype=int)
        elif (sdir / "grid.geojson").exists():
            grid_lons, grid_lats, _ = self._coords_from_geojson(sdir / "grid.geojson")

        if (sdir / "labels.npy").exists():
            labels = np.load(sdir / "labels.npy")

        centroids_npz = sdir / "centroids.npz"
        if centroids_npz.exists():
            z = np.load(centroids_npz)
            centroid_lons = z["lons"]
            centroid_lats = z["lats"]
            if "damages" in z.files:
                centroid_damages = z["damages"]
        else:
            cpath = sdir / "centroids_damage.geojson"
            if not cpath.exists():
                cpath = sdir / "centroids.geojson"
            if cpath.exists():
                centroid_lons, centroid_lats, cprops = self._coords_from_geojson(cpath)
                if cprops and "damage" in cprops[0]:
                    centroid_damages = np.asarray(
                        [float(p.get("damage", np.nan)) for p in cprops], dtype=float
                    )

        if (sdir / "idw.npy").exists():
            idw_raster = np.load(sdir / "idw.npy")
            bounds = (state.meta.get("idw") or {}).get("bounds")
            if bounds:
                idw_bounds = bounds

        self._mem[session_id] = {
            "state": state,
            "geom": geom,
            "samples": samples,
            "grid_lons": grid_lons,
            "grid_lats": grid_lats,
            "labels": labels,
            "centroid_lons": centroid_lons,
            "centroid_lats": centroid_lats,
            "centroid_damages": centroid_damages,
            "idw_raster": idw_raster,
            "idw_bounds": idw_bounds,
        }

    def create(self, lot_id: str, mode: str, grid_res_m: float) -> SessionState:
        feature = catalog.get_lot_feature(lot_id)
        summary = catalog.get_lot_summary(lot_id)
        if feature is None or summary is None:
            raise ValueError(f"Lote no encontrado: {lot_id}")

        geom = geom_from_geojson(feature["geometry"])
        area = float(summary.get("area_ha") or area_ha_wgs84(geom))
        k = min_samples_for_area_ha(area)
        session_id = uuid.uuid4().hex[:12]
        state = SessionState(
            session_id=session_id,
            lot_id=str(lot_id),
            mode=mode,
            grid_res_m=float(grid_res_m),
            area_ha=area,
            k=k,
            status="created",
            steps_done=[],
            meta={
                "name": summary.get("name"),
                "sample_count": summary.get("sample_count"),
                "crop_name": summary.get("crop_name"),
            },
        )
        sdir = self._dir(session_id)
        with (sdir / "lot.geojson").open("w", encoding="utf-8") as f:
            json.dump(
                {"type": "FeatureCollection", "features": [polygon_feature(geom, feature["properties"])]},
                f,
            )
        samples = catalog.samples_for_lot(lot_id)
        samples.to_parquet(sdir / "samples.parquet", index=False)
        sample_gj = points_to_feature_collection(
            samples["lon"].to_numpy(),
            samples["lat"].to_numpy(),
            [
                {
                    "id": int(i),
                    "damage": float(samples.iloc[i]["damage"]),
                    "lot_name": str(samples.iloc[i].get("lot_name") or ""),
                    "nro_siniestro": str(samples.iloc[i].get("nro_siniestro") or ""),
                    "orden_trabajo": str(samples.iloc[i].get("orden_trabajo") or ""),
                }
                for i in range(len(samples))
            ],
        )
        with (sdir / "samples.geojson").open("w", encoding="utf-8") as f:
            json.dump(sample_gj, f)

        self._mem[session_id] = {
            "state": state,
            "geom": geom,
            "samples": samples,
            "grid_lons": None,
            "grid_lats": None,
            "labels": None,
            "centroid_lons": None,
            "centroid_lats": None,
            "centroid_damages": None,
            "idw_raster": None,
            "idw_bounds": None,
        }
        self._save_meta(session_id)
        return state

    def get(self, session_id: str) -> dict[str, Any]:
        if session_id not in self._mem:
            self._hydrate(session_id)
        return self._mem[session_id]

    def state(self, session_id: str) -> SessionState:
        return self.get(session_id)["state"]

    def artifact_path(self, session_id: str, name: str) -> Path:
        return settings.sessions_dir / session_id / name

    def step_grid(self, session_id: str) -> dict[str, Any]:
        sess = self.get(session_id)
        geom = sess["geom"]
        res = sess["state"].grid_res_m
        lons, lats, meta = grid_points_in_polygon(geom, res)
        sess["grid_lons"] = lons
        sess["grid_lats"] = lats
        outline = grid_outline_geojson(geom, res)
        grid_gj = points_to_feature_collection(
            lons, lats, [{"id": int(i)} for i in range(len(lons))]
        )
        # subsample for map if huge
        max_pts = 5000
        if len(lons) > max_pts:
            idx = np.linspace(0, len(lons) - 1, max_pts).astype(int)
            grid_gj = points_to_feature_collection(
                lons[idx], lats[idx], [{"id": int(i)} for i in idx]
            )
        sdir = self._dir(session_id)
        with (sdir / "grid.geojson").open("w", encoding="utf-8") as f:
            json.dump(grid_gj, f)
        with (sdir / "grid_outline.geojson").open("w", encoding="utf-8") as f:
            json.dump(outline, f)
        with (sdir / "grid_meta.json").open("w", encoding="utf-8") as f:
            json.dump(meta, f)
        st = sess["state"]
        if "grid" not in st.steps_done:
            st.steps_done.append("grid")
        st.status = "grid"
        self._save_meta(session_id)
        self._persist_arrays(session_id)
        return {
            "n_cells": meta["n_cells"],
            "grid_res_m": res,
            "k": st.k,
            "area_ha": st.area_ha,
            "grid_url": f"/api/sessions/{session_id}/artifacts/grid.geojson",
            "outline_url": f"/api/sessions/{session_id}/artifacts/grid_outline.geojson",
            "lot_url": f"/api/sessions/{session_id}/artifacts/lot.geojson",
        }

    def step_kmeans(self, session_id: str) -> dict[str, Any]:
        sess = self.get(session_id)
        if sess["grid_lons"] is None:
            self.step_grid(session_id)
        st = sess["state"]
        result = run_kmeans(sess["grid_lons"], sess["grid_lats"], st.k, sess["geom"])
        sess["labels"] = result["labels"]
        sess["centroid_lons"] = result["centroid_lons"]
        sess["centroid_lats"] = result["centroid_lats"]
        sdir = self._dir(session_id)
        for name, key in (
            ("centroids.geojson", "centroids_geojson"),
            ("clusters.geojson", "clusters_geojson"),
            ("grid_labeled.geojson", "grid_labeled_geojson"),
        ):
            with (sdir / name).open("w", encoding="utf-8") as f:
                json.dump(result[key], f)
        if "kmeans" not in st.steps_done:
            st.steps_done.append("kmeans")
        st.status = "kmeans"
        self._save_meta(session_id)
        self._persist_arrays(session_id)
        return {
            "k": result["k"],
            "centroids_url": f"/api/sessions/{session_id}/artifacts/centroids.geojson",
            "clusters_url": f"/api/sessions/{session_id}/artifacts/clusters.geojson",
            "grid_labeled_url": f"/api/sessions/{session_id}/artifacts/grid_labeled.geojson",
        }

    def step_samples(self, session_id: str) -> dict[str, Any]:
        sess = self.get(session_id)
        st = sess["state"]
        samples = sess["samples"]
        if "samples" not in st.steps_done:
            st.steps_done.append("samples")
        st.status = "samples"
        self._save_meta(session_id)
        return {
            "n_samples": int(len(samples)),
            "damage_stats": summary_stats(samples["damage"].to_numpy(dtype=float)),
            "samples_url": f"/api/sessions/{session_id}/artifacts/samples.geojson",
        }

    def step_assign(self, session_id: str) -> dict[str, Any]:
        sess = self.get(session_id)
        if sess["centroid_lons"] is None:
            self.step_kmeans(session_id)
        samples = sess["samples"]
        result = assign_nearest_damage(
            sess["centroid_lons"],
            sess["centroid_lats"],
            samples["lon"].to_numpy(dtype=float),
            samples["lat"].to_numpy(dtype=float),
            samples["damage"].to_numpy(dtype=float),
            sess["geom"],
        )
        sess["centroid_damages"] = result["damages"]
        sdir = self._dir(session_id)
        with (sdir / "centroids_damage.geojson").open("w", encoding="utf-8") as f:
            json.dump(result["centroids_geojson"], f)
        st = sess["state"]
        if "assign" not in st.steps_done:
            st.steps_done.append("assign")
        st.status = "assign"
        self._save_meta(session_id)
        self._persist_arrays(session_id)
        return {
            "damage_mean": result["damage_mean"],
            "damage_min": result["damage_min"],
            "damage_max": result["damage_max"],
            "centroids_url": f"/api/sessions/{session_id}/artifacts/centroids_damage.geojson",
        }

    def step_idw(self, session_id: str, params: IdwParams) -> dict[str, Any]:
        sess = self.get(session_id)
        if sess["centroid_damages"] is None:
            self.step_assign(session_id)
        sdir = self._dir(session_id)
        result = idw_interpolate(
            sess["geom"],
            sess["centroid_lons"],
            sess["centroid_lats"],
            sess["centroid_damages"],
            res_m=sess["state"].grid_res_m,
            power=params.power,
            scale_mode=params.scale_mode,
            symbology=params.symbology,
            out_png=sdir / "idw.png",
            out_npy=sdir / "idw.npy",
        )
        sess["idw_raster"] = result["raster"]
        sess["idw_bounds"] = result["bounds"]
        # drop heavy array from returned payload
        payload = {k: v for k, v in result.items() if k != "raster"}
        payload["png_url"] = f"/api/sessions/{session_id}/artifacts/idw.png"
        st = sess["state"]
        if "idw" not in st.steps_done:
            st.steps_done.append("idw")
        st.status = "idw"
        st.meta["idw"] = {
            "vmin": result["vmin"],
            "vmax": result["vmax"],
            "mean_damage": result["mean_damage"],
            "bounds": result["bounds"],
            "scale_mode": params.scale_mode,
            "symbology": params.symbology,
            "power": params.power,
        }
        self._save_meta(session_id)
        self._persist_arrays(session_id)
        return payload

    def step_zones(self, session_id: str, params: ZoneParams) -> dict[str, Any]:
        sess = self.get(session_id)
        if sess["idw_raster"] is None:
            self.step_idw(session_id, IdwParams())
        sdir = self._dir(session_id)
        result = zonify_raster(
            sess["idw_raster"],
            scale_mode=params.scale_mode,
            symbology=params.symbology,
            out_png=sdir / "zones.png",
            bounds=sess["idw_bounds"],
        )
        result["png_url"] = f"/api/sessions/{session_id}/artifacts/zones.png"
        st = sess["state"]
        if "zones" not in st.steps_done:
            st.steps_done.append("zones")
        st.status = "zones"
        st.meta["zones"] = result
        self._save_meta(session_id)
        return result

    def step_compare(self, session_id: str) -> dict[str, Any]:
        sess = self.get(session_id)
        if sess["centroid_damages"] is None:
            self.step_assign(session_id)
        if sess["idw_raster"] is None:
            self.step_idw(session_id, IdwParams())
        if sess["labels"] is None:
            self.step_kmeans(session_id)
        st = sess["state"]
        cmp, ambiente_labels = compare_methods(
            sess["samples"]["damage"].to_numpy(dtype=float),
            sess["idw_raster"],
            sess["labels"],
            sess["centroid_damages"],
            st.k,
            st.area_ha,
            st.grid_res_m,
        )
        sdir = self._dir(session_id)
        ambientes_fc = cluster_hulls(sess["grid_lons"], sess["grid_lats"], ambiente_labels)
        for feat in ambientes_fc.get("features", []):
            props = feat.setdefault("properties", {})
            a = int(props.get("cluster", 0))
            props["ambiente"] = a
            props["label"] = f"A{a + 1}"
            if a < len(cmp.ambientes):
                props["damage"] = cmp.ambientes[a].damage
                props["area_ha"] = cmp.ambientes[a].area_ha
                props["weight_pct"] = cmp.ambientes[a].weight_pct
        with (sdir / "ambientes.geojson").open("w", encoding="utf-8") as f:
            json.dump(ambientes_fc, f)
        if "compare" not in st.steps_done:
            st.steps_done.append("compare")
        st.status = "compare"
        st.meta["compare"] = cmp.model_dump()
        self._save_meta(session_id)
        return {
            **cmp.model_dump(),
            "sample_stats": summary_stats(sess["samples"]["damage"].to_numpy(dtype=float)),
            "idw_stats": summary_stats(sess["idw_raster"]),
            "idw_png_url": f"/api/sessions/{session_id}/artifacts/idw.png",
            "zones_png_url": f"/api/sessions/{session_id}/artifacts/zones.png",
            "samples_url": f"/api/sessions/{session_id}/artifacts/samples.geojson",
            "centroids_url": f"/api/sessions/{session_id}/artifacts/centroids_damage.geojson",
            "clusters_url": f"/api/sessions/{session_id}/artifacts/clusters.geojson",
            "ambientes_url": f"/api/sessions/{session_id}/artifacts/ambientes.geojson",
            "lot_url": f"/api/sessions/{session_id}/artifacts/lot.geojson",
            "bounds": sess["idw_bounds"],
        }

    def run_all(self, session_id: str, idw: IdwParams | None = None, zones: ZoneParams | None = None) -> dict[str, Any]:
        idw = idw or IdwParams()
        zones = zones or ZoneParams()
        grid = self.step_grid(session_id)
        kmeans = self.step_kmeans(session_id)
        samples = self.step_samples(session_id)
        assign = self.step_assign(session_id)
        idw_res = self.step_idw(session_id, idw)
        zones_res = self.step_zones(session_id, zones)
        compare = self.step_compare(session_id)
        return {
            "grid": grid,
            "kmeans": kmeans,
            "samples": samples,
            "assign": assign,
            "idw": idw_res,
            "zones": zones_res,
            "compare": compare,
        }


sessions = SessionStore()
