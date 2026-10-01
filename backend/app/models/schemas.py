from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class LotSummary(BaseModel):
    lot_id: str
    name: str
    farm_name: str | None = None
    crop_name: str | None = None
    area_ha: float
    sample_count: int = 0
    bbox: list[float] | None = None  # [minx, miny, maxx, maxy]
    campaign: str | None = None


class SessionCreate(BaseModel):
    lot_id: str
    mode: Literal["learning", "production"] = "learning"
    grid_res_m: float = Field(default=10.0, ge=1.0, le=50.0)


class IdwParams(BaseModel):
    power: float = Field(default=2.0, ge=0.5, le=5.0)
    scale_mode: Literal["fixed", "adaptive"] = "fixed"
    symbology: Literal["continuous", "discrete"] = "continuous"


class ZoneParams(BaseModel):
    scale_mode: Literal["fixed", "adaptive"] = "fixed"
    symbology: Literal["continuous", "discrete"] = "discrete"


class ClusterContribution(BaseModel):
    cluster: int
    label: str
    n_cells: int
    area_ha: float
    weight: float
    weight_pct: float
    damage: float
    contribution: float
    contribution_pct: float


class CompareResult(BaseModel):
    """traditional_mean = promedio simple de muestras;
    weighted_mean = ponderado por área de clusters K-Means (k);
    weighted_ambientes_mean = re-KMeans por similitud de daño → menos ambientes;
    directed_mean = media del raster IDW (referencia espacial continua).
    """

    traditional_mean: float
    weighted_mean: float
    weighted_ambientes_mean: float = 0.0
    directed_mean: float
    abs_diff: float
    pct_diff: float
    abs_diff_ambientes: float = 0.0
    pct_diff_ambientes: float = 0.0
    abs_diff_k_vs_ambientes: float = 0.0
    pct_diff_k_vs_ambientes: float = 0.0
    abs_diff_idw: float = 0.0
    pct_diff_idw: float = 0.0
    n_samples: int
    k: int
    k_ambientes: int = 0
    area_ha: float
    grid_res_m: float
    clusters: list[ClusterContribution] = Field(default_factory=list)
    ambientes: list[ClusterContribution] = Field(default_factory=list)
    cluster_to_ambiente: dict[int, int] = Field(default_factory=dict)
    ambientes_silhouette: float | None = None


class SessionState(BaseModel):
    session_id: str
    lot_id: str
    mode: str
    grid_res_m: float
    area_ha: float
    k: int
    status: str = "created"
    steps_done: list[str] = Field(default_factory=list)
    meta: dict[str, Any] = Field(default_factory=dict)


class PipelineStepResult(BaseModel):
    session_id: str
    step: str
    ok: bool
    message: str
    data: dict[str, Any] = Field(default_factory=dict)
