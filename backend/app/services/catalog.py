from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from shapely.geometry import Point, mapping as shp_mapping
from shapely.strtree import STRtree

from app.core.config import settings
from app.core.sample_rules import min_samples_for_area_ha
from app.services.geo_utils import area_ha_wgs84

# Marcos Juárez approximate bbox for fast filtering
MJ_BBOX = (-62.9, -34.0, -61.7, -32.0)


def _safe_float(v: Any) -> float | None:
    try:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return None
        f = float(v)
        if not np.isfinite(f):
            return None
        return f
    except (TypeError, ValueError):
        return None


def load_claims_dataframe(path: Path | None = None) -> pd.DataFrame:
    """
    Load claim samples.

    Hierarchy: siniestro → orden de trabajo → lote → muestras.
    If several work orders touch the same lot, only the first WO is kept
    (earliest assigned/event date, then lowest workOrderId). That filter is
    applied after spatial join in build_or_load_index; here we normalize fields.
    """
    path = path or settings.claims_path
    df = pd.read_csv(path, sep=";", low_memory=False)

    lat = _numeric_col(df, ["ros_lat_muestra", "gpsPointSampleX"])
    lon = _numeric_col(df, ["ros_lon_muestra", "gpsPointSampleY"])
    dmg = _numeric_col(df, ["ros_dano_por_muestra", "calculatedDamagePercentage", "damagePercentage"])

    nro = _first_series(df, ["NroSiniestro", "NroReferencia_modo", "referenceId", "lossID"])
    wo = _first_series(df, ["workOrderId", "_key_workorder"])
    lot_name = _first_series(df, ["ros_lote_nombre", "lotName", "ros_lote__lotName"])
    muestra = _first_series(df, ["muestraId", "ros_muestra__muestra_index"])
    cultivo = _first_series(df, ["nova_lote__harvest", "DescCultivo_modo", "CropName"])

    assigned = _parse_claim_date(_first_series(df, ["ros_ot__assignedDate", "fecha_tasacion_nova", "occurDate"]))
    event = _parse_claim_date(_first_series(df, ["fecha_evento_nova", "occurDate", "fecha_denuncia_nova"]))

    origen = df.get("origen_muestra") if "origen_muestra" in df.columns else None
    mask = lat.between(-90, 90) & lon.between(-180, 180) & dmg.notna()
    if origen is not None:
        mask = mask & (origen.astype(str) != "top_level")

    out = pd.DataFrame(
        {
            "lat": lat[mask].astype(float),
            "lon": lon[mask].astype(float),
            "damage": dmg[mask].astype(float).clip(0, 100),
            "lot_name": _as_clean_str(lot_name[mask]),
            "nro_siniestro": _as_clean_str(nro[mask]),
            "orden_trabajo": _as_clean_str(wo[mask]),
            "muestra_id": _as_clean_str(muestra[mask]),
            "cultivo": _as_clean_str(cultivo[mask]),
            "fecha_orden": assigned[mask].values,
            "fecha_evento": event[mask].values,
        }
    )
    out = out.drop_duplicates(
        subset=["lat", "lon", "damage", "nro_siniestro", "orden_trabajo", "muestra_id"]
    ).reset_index(drop=True)
    return out


def _first_series(df: pd.DataFrame, names: list[str]) -> pd.Series:
    for name in names:
        if name in df.columns:
            return df[name]
    return pd.Series([None] * len(df), index=df.index)


def _numeric_col(df: pd.DataFrame, names: list[str]) -> pd.Series:
    for name in names:
        if name in df.columns:
            col = pd.to_numeric(df[name], errors="coerce")
            if not col.isna().all():
                return col
    return pd.Series(np.nan, index=df.index, dtype=float)


def _as_clean_str(series: pd.Series) -> pd.Series:
    s = series.map(lambda v: "" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v))
    return s.str.replace(r"\.0$", "", regex=True).replace({"nan": "", "None": "", "<NA>": ""})


def _parse_claim_date(series: pd.Series) -> pd.Series:
    """Parse mixed claim date strings to datetime64 (NaT on failure)."""
    raw = series.astype(str).str.strip()
    # "18/02/202000:00:00" → "18/02/2020 00:00:00"
    raw = raw.str.replace(r"(\d{2}/\d{2}/\d{4})(\d{2}:\d{2}:\d{2})", r"\1 \2", regex=True)
    # "2020-02-1616:00:00" → "2020-02-16 16:00:00"
    raw = raw.str.replace(r"(\d{4}-\d{2}-\d{2})(\d{2}:\d{2}:\d{2})", r"\1 \2", regex=True)
    iso = raw.str.match(r"^\d{4}-\d{2}-\d{2}")
    parsed = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
    if iso.any():
        parsed.loc[iso] = pd.to_datetime(raw.loc[iso], errors="coerce")
    if (~iso).any():
        parsed.loc[~iso] = pd.to_datetime(raw.loc[~iso], dayfirst=True, errors="coerce")
    return parsed


def prefer_first_work_order(claims: pd.DataFrame, sample_idxs: list[int]) -> list[int]:
    """
    Among samples that fall in one lot, keep only those from the first work order.
    First = earliest fecha_orden / fecha_evento, then lowest orden_trabajo id.
    """
    if not sample_idxs:
        return []
    sub = claims.iloc[sample_idxs].copy()
    sub["_orig_idx"] = sample_idxs
    if "orden_trabajo" not in sub.columns:
        return sample_idxs
    sub["orden_trabajo"] = sub["orden_trabajo"].astype(str).replace({"nan": "", "None": ""})
    sub = sub[sub["orden_trabajo"] != ""]
    if sub.empty:
        return sample_idxs
    if sub["orden_trabajo"].nunique() <= 1:
        return sub["_orig_idx"].astype(int).tolist()

    wo_rank = (
        sub.groupby("orden_trabajo", as_index=False)
        .agg(
            fecha_orden=("fecha_orden", "min"),
            fecha_evento=("fecha_evento", "min"),
        )
        .sort_values(
            by=["fecha_orden", "fecha_evento", "orden_trabajo"],
            ascending=True,
            na_position="last",
        )
    )
    first_wo = str(wo_rank.iloc[0]["orden_trabajo"])
    return sub.loc[sub["orden_trabajo"] == first_wo, "_orig_idx"].astype(int).tolist()


def _lot_id_from_props(props: dict[str, Any], idx: int) -> str:
    for key in ("FarmShapeI", "OBJECTID", "fid"):
        if props.get(key) is not None:
            try:
                return str(int(float(props[key])))
            except (TypeError, ValueError):
                return str(props[key])
    return f"idx-{idx}"


def build_or_load_index(force: bool = False) -> dict[str, Any]:
    """
    Build a lightweight index of lots that intersect claim samples (Marcos Juárez focus).
    Saves GeoJSON of matching lots + samples parquet for quick API responses.
    """
    settings.index_dir.mkdir(parents=True, exist_ok=True)
    lots_out = settings.index_dir / "lots_with_samples.geojson"
    samples_out = settings.index_dir / "samples.parquet"
    meta_out = settings.index_dir / "index_meta.json"

    if not force and lots_out.exists() and samples_out.exists() and meta_out.exists():
        with meta_out.open(encoding="utf-8") as f:
            return json.load(f)

    if not settings.claims_path.exists() or not settings.lots_path.exists():
        return {"ready": False, "lot_count": 0, "message": "Datos raw no encontrados"}

    print("[AGRO] Loading claims...")
    claims = load_claims_dataframe()
    claims.to_parquet(samples_out, index=False)
    print(f"[AGRO] Claims with coords: {len(claims)}")

    import geopandas as gpd

    print("[AGRO] Loading lots in Marcos Juárez bbox...")
    minx, miny, maxx, maxy = MJ_BBOX
    lots = gpd.read_file(settings.lots_path, bbox=MJ_BBOX)
    lots = lots.reset_index(drop=True)
    print(f"[AGRO] Lots in bbox: {len(lots)}")

    points = [Point(xy) for xy in zip(claims["lon"].tolist(), claims["lat"].tolist())]
    tree = STRtree(lots.geometry.values)
    lot_sample_idxs: dict[int, list[int]] = {}
    print("[AGRO] Spatial join samples -> lots...")
    for i, pt in enumerate(points):
        candidates = tree.query(pt)
        for li in candidates:
            li = int(li)
            geom = lots.geometry.iloc[li]
            if geom.contains(pt) or geom.touches(pt):
                lot_sample_idxs.setdefault(li, []).append(i)

    # Misma lote + varias OT → quedarse con la primera OT (fecha / id)
    for li, idxs in list(lot_sample_idxs.items()):
        lot_sample_idxs[li] = prefer_first_work_order(claims, idxs)
    lot_sample_counts = {li: len(idxs) for li, idxs in lot_sample_idxs.items()}

    keep_idx = [i for i, c in lot_sample_counts.items() if c >= 3]
    keep_idx = sorted(keep_idx, key=lambda i: lot_sample_counts[i], reverse=True)[:200]
    print(f"[AGRO] Lots with >=3 samples: {len(keep_idx)}")

    features = []
    summaries = []
    for li in keep_idx:
        row = lots.iloc[li]
        props = {k: (None if pd.isna(v) else v) for k, v in row.drop(labels="geometry").items()}
        geom = row.geometry
        lid = _lot_id_from_props(props, li)
        area = _safe_float(props.get("ShapeArea")) or area_ha_wgs84(geom)
        name = str(props.get("Name") or f"Lote {lid}")
        scount = lot_sample_counts[li]
        minx_, miny_, maxx_, maxy_ = geom.bounds
        summaries.append(
            {
                "lot_id": lid,
                "name": name,
                "farm_name": props.get("FarmName"),
                "crop_name": props.get("CropName"),
                "area_ha": float(area),
                "sample_count": scount,
                "bbox": [minx_, miny_, maxx_, maxy_],
                "campaign": props.get("Campaign"),
                "min_samples_rule": min_samples_for_area_ha(float(area)),
                "sample_indices": lot_sample_idxs[li],
            }
        )
        features.append(
            {
                "type": "Feature",
                "id": lid,
                "properties": {
                    **{
                        k: props[k]
                        for k in (
                            "Name",
                            "FarmName",
                            "CropName",
                            "ShapeArea",
                            "Campaign",
                            "FarmShapeI",
                            "OBJECTID",
                        )
                        if k in props
                    },
                    "lot_id": lid,
                    "area_ha": float(area),
                    "sample_count": scount,
                },
                "geometry": shp_mapping(geom),
            }
        )

    fc = {"type": "FeatureCollection", "features": features}
    with lots_out.open("w", encoding="utf-8") as f:
        json.dump(fc, f)

    with (settings.index_dir / "lot_summaries.json").open("w", encoding="utf-8") as f:
        json.dump(summaries, f)

    meta = {
        "ready": True,
        "lot_count": len(summaries),
        "sample_count": int(len(claims)),
        "lots_path": str(lots_out),
        "samples_path": str(samples_out),
        "summaries_path": str(settings.index_dir / "lot_summaries.json"),
    }
    with meta_out.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return meta


class DataCatalog:
    def __init__(self) -> None:
        self.meta: dict[str, Any] = {}
        self.summaries: list[dict[str, Any]] = []
        self._lots_by_id: dict[str, dict[str, Any]] = {}
        self._claims: pd.DataFrame | None = None

    def ensure_ready(self, force: bool = False) -> dict[str, Any]:
        self.meta = build_or_load_index(force=force)
        if self.meta.get("ready"):
            with Path(self.meta["summaries_path"]).open(encoding="utf-8") as f:
                self.summaries = json.load(f)
            with Path(self.meta["lots_path"]).open(encoding="utf-8") as f:
                fc = json.load(f)
            self._lots_by_id = {str(ft["properties"]["lot_id"]): ft for ft in fc["features"]}
            self._claims = pd.read_parquet(self.meta["samples_path"])
        return self.meta

    def list_lots(self, q: str | None = None, with_samples_only: bool = True) -> list[dict[str, Any]]:
        items = self.summaries
        if q:
            ql = q.lower()
            items = [
                s
                for s in items
                if ql in str(s.get("name", "")).lower()
                or ql in str(s.get("farm_name", "")).lower()
                or ql in str(s.get("lot_id", "")).lower()
            ]
        return items

    def get_lot_feature(self, lot_id: str) -> dict[str, Any] | None:
        return self._lots_by_id.get(str(lot_id))

    def get_lot_summary(self, lot_id: str) -> dict[str, Any] | None:
        for s in self.summaries:
            if str(s["lot_id"]) == str(lot_id):
                return s
        return None

    def samples_for_lot(self, lot_id: str) -> pd.DataFrame:
        summary = self.get_lot_summary(lot_id)
        if summary is None or self._claims is None:
            return pd.DataFrame()
        idxs = summary.get("sample_indices") or []
        return self._claims.iloc[idxs].copy().reset_index(drop=True)


catalog = DataCatalog()
