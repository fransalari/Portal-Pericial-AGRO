from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.models.schemas import IdwParams, SessionCreate, ZoneParams
from app.services.catalog import catalog
from app.services.geo_utils import points_to_feature_collection
from app.services.session_store import sessions

router = APIRouter(prefix="/api")


class InspectBody(BaseModel):
    grid_res_m: float = Field(default=10.0, ge=1.0, le=50.0)
    method: str = Field(default="both")  # traditional | directed | both


def _samples_geojson(lot_id: str) -> dict:
    samples = catalog.samples_for_lot(lot_id)
    if samples is None or len(samples) == 0:
        return {"type": "FeatureCollection", "features": []}
    props = []
    for i in range(len(samples)):
        row = samples.iloc[i]
        props.append(
            {
                "id": int(i),
                "damage": float(row["damage"]),
                "lot_name": str(row.get("lot_name") or ""),
                "nro_siniestro": str(row.get("nro_siniestro") or ""),
                "orden_trabajo": str(row.get("orden_trabajo") or ""),
                "muestra_id": str(row.get("muestra_id") or ""),
            }
        )
    return points_to_feature_collection(
        samples["lon"].to_numpy(dtype=float),
        samples["lat"].to_numpy(dtype=float),
        props,
    )


def _dedupe_lots(lots: list[dict]) -> list[dict]:
    best: dict[str, dict] = {}
    for lot in lots:
        lid = str(lot["lot_id"])
        prev = best.get(lid)
        if prev is None or int(lot.get("sample_count") or 0) > int(prev.get("sample_count") or 0):
            best[lid] = lot
    return sorted(best.values(), key=lambda x: int(x.get("sample_count") or 0), reverse=True)


@router.get("/health")
def health():
    return {
        "ok": True,
        "app": "portal-pericial",
        "index_ready": bool(catalog.meta.get("ready")),
    }


@router.get("/cases")
def list_cases(q: str | None = None, limit: int = Query(default=40, ge=1, le=200)):
    """Casos asignados al tasador (= lotes únicos con muestras)."""
    lots = _dedupe_lots(catalog.list_lots(q=q))[:limit]
    cases = []
    for i, lot in enumerate(lots):
        cases.append(
            {
                "case_id": f"CASO-{lot['lot_id']}",
                "lot_id": str(lot["lot_id"]),
                "status": "asignado" if i % 3 else "pendiente_visita",
                "priority": "alta" if lot.get("sample_count", 0) >= 20 else "media",
                "event": "Granizo / clima",
                "assignee": "Tasador · Zona Marcos Juárez",
                "sla_hours": 72,
                "lot": lot,
            }
        )
    return {"count": len(cases), "cases": cases}


@router.get("/cases/{lot_id}")
def get_case(lot_id: str):
    summary = catalog.get_lot_summary(lot_id)
    feature = catalog.get_lot_feature(lot_id)
    if not summary or not feature:
        raise HTTPException(404, "Caso / lote no encontrado")
    samples = catalog.samples_for_lot(lot_id)
    nro = ""
    wo = ""
    if len(samples):
        nro = str(samples.iloc[0].get("nro_siniestro") or "")
        wo = str(samples.iloc[0].get("orden_trabajo") or "")
    return {
        "case_id": f"CASO-{lot_id}",
        "lot_id": lot_id,
        "summary": summary,
        "feature": feature,
        "samples_geojson": _samples_geojson(lot_id),
        "nro_siniestro": nro,
        "orden_trabajo": wo,
        "n_samples": int(len(samples)),
        "workflow": [
            "Denuncia ingresada",
            "Análisis de zona y cultivo",
            "Asignación a tasador",
            "Visita y muestreo",
            "Cierre de tasación",
        ],
    }


@router.post("/cases/{lot_id}/inspect")
def inspect_case(lot_id: str, body: InspectBody | None = None):
    """
    Inspección completa:
    - traditional → muestras + promedio simple
    - directed|both → grilla, K-Means, asignación, IDW, zonas, comparación
    """
    body = body or InspectBody()
    try:
        state = sessions.create(lot_id, "production", body.grid_res_m)
    except ValueError as e:
        raise HTTPException(404, str(e)) from e

    sid = state.session_id
    base_urls = {
        "lot_url": f"/api/sessions/{sid}/artifacts/lot.geojson",
        "samples_url": f"/api/sessions/{sid}/artifacts/samples.geojson",
    }

    if body.method == "traditional":
        samples = sessions.step_samples(sid)
        sess = sessions.get(sid)
        import numpy as np

        damages = sess["samples"]["damage"].to_numpy(dtype=float)
        traditional = float(np.nanmean(damages)) if len(damages) else float("nan")
        return {
            "session_id": sid,
            "method": "traditional",
            "state": sessions.state(sid),
            "traditional_mean": round(traditional, 4),
            "n_samples": int(len(damages)),
            "samples": samples,
            **base_urls,
        }

    result = sessions.run_all(sid)
    compare = result["compare"]
    return {
        "session_id": sid,
        "method": body.method,
        "state": sessions.state(sid),
        "traditional_mean": compare.get("traditional_mean"),
        "weighted_mean": compare.get("weighted_mean"),
        "weighted_ambientes_mean": compare.get("weighted_ambientes_mean"),
        "directed_mean": compare.get("directed_mean"),
        "abs_diff": compare.get("abs_diff"),
        "abs_diff_ambientes": compare.get("abs_diff_ambientes"),
        "abs_diff_k_vs_ambientes": compare.get("abs_diff_k_vs_ambientes"),
        "k": compare.get("k"),
        "k_ambientes": compare.get("k_ambientes"),
        "clusters": compare.get("clusters"),
        "ambientes": compare.get("ambientes"),
        "pipeline": {
            "grid": result.get("grid"),
            "kmeans": result.get("kmeans"),
            "assign": result.get("assign"),
            "idw": result.get("idw"),
            "zones": result.get("zones"),
        },
        "compare": compare,
        **base_urls,
        "centroids_url": f"/api/sessions/{sid}/artifacts/centroids_damage.geojson",
        "clusters_url": f"/api/sessions/{sid}/artifacts/clusters.geojson",
        "ambientes_url": f"/api/sessions/{sid}/artifacts/ambientes.geojson",
        "idw_png_url": f"/api/sessions/{sid}/artifacts/idw.png",
        "zones_png_url": f"/api/sessions/{sid}/artifacts/zones.png",
        "bounds": compare.get("bounds") or (result.get("idw") or {}).get("bounds"),
    }


@router.post("/sessions")
def create_session(body: SessionCreate):
    try:
        return sessions.create(body.lot_id, body.mode, body.grid_res_m)
    except ValueError as e:
        raise HTTPException(404, str(e)) from e


@router.get("/sessions/{session_id}")
def get_session(session_id: str):
    try:
        return sessions.state(session_id)
    except KeyError as e:
        raise HTTPException(404, str(e)) from e


@router.post("/sessions/{session_id}/run")
def run_pipeline(session_id: str, idw: IdwParams | None = None, zones: ZoneParams | None = None):
    try:
        return sessions.run_all(session_id, idw=idw, zones=zones)
    except (KeyError, ValueError) as e:
        raise HTTPException(400, str(e)) from e


@router.get("/sessions/{session_id}/compare")
def step_compare(session_id: str):
    try:
        return sessions.step_compare(session_id)
    except KeyError as e:
        raise HTTPException(404, str(e)) from e


@router.get("/sessions/{session_id}/artifacts/{name}")
def get_artifact(session_id: str, name: str):
    allowed = {
        "lot.geojson",
        "samples.geojson",
        "grid.geojson",
        "grid_outline.geojson",
        "grid_meta.json",
        "centroids.geojson",
        "centroids_damage.geojson",
        "clusters.geojson",
        "ambientes.geojson",
        "grid_labeled.geojson",
        "idw.png",
        "zones.png",
        "state.json",
    }
    if name not in allowed:
        raise HTTPException(404, "Artefacto no permitido")
    path = sessions.artifact_path(session_id, name)
    if not path.exists():
        raise HTTPException(404, "Artefacto no encontrado")
    try:
        sessions.get(session_id)
    except KeyError:
        pass
    media = "image/png" if name.endswith(".png") else "application/json"
    return FileResponse(path, media_type=media)
