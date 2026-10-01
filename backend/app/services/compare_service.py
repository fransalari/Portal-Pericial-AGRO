from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from app.models.schemas import ClusterContribution, CompareResult


def cluster_contributions(
    labels: np.ndarray,
    centroid_damages: np.ndarray,
    grid_res_m: float,
    *,
    label_prefix: str = "C",
) -> tuple[list[ClusterContribution], float]:
    """
    Area of each cluster = n_cells × cell_area (equal-area operational grid).
    Weighted mean = Σ (damage_c × area_c) / Σ area_c.
    """
    labels = np.asarray(labels, dtype=int)
    damages = np.asarray(centroid_damages, dtype=float)
    cell_ha = float(grid_res_m) ** 2 / 10_000.0
    if len(damages) == 0 or len(labels) == 0:
        return [], float("nan")

    ids = sorted(int(x) for x in np.unique(labels))
    rows: list[ClusterContribution] = []
    areas: list[float] = []
    for idx, c in enumerate(ids):
        n_cells = int(np.sum(labels == c))
        area_ha = n_cells * cell_ha
        areas.append(area_ha)
        dmg = float(damages[c]) if c < len(damages) and np.isfinite(damages[c]) else 0.0
        rows.append(
            ClusterContribution(
                cluster=int(c),
                label=f"{label_prefix}{idx + 1}",
                n_cells=n_cells,
                area_ha=round(area_ha, 4),
                weight=0.0,
                weight_pct=0.0,
                damage=round(dmg, 4),
                contribution=0.0,
                contribution_pct=0.0,
            )
        )

    total_area = float(sum(areas))
    if total_area <= 0:
        return rows, float("nan")

    weighted = 0.0
    for i, row in enumerate(rows):
        w = areas[i] / total_area
        contrib = row.damage * w
        weighted += contrib
        rows[i] = row.model_copy(
            update={
                "weight": round(w, 6),
                "weight_pct": round(w * 100.0, 2),
                "contribution": round(contrib, 4),
            }
        )

    if weighted != 0 and np.isfinite(weighted):
        rows = [
            r.model_copy(update={"contribution_pct": round(r.contribution / weighted * 100.0, 2)})
            for r in rows
        ]

    return rows, float(weighted)


def reduce_ambientes_by_damage_similarity(
    labels: np.ndarray,
    centroid_damages: np.ndarray,
    grid_res_m: float,
) -> dict[str, Any]:
    """
    Re-run K-Means on centroid damages (1D) to merge clusters with similar damage
    into fewer ambientes, then recompute area-weighted lot damage.
    """
    labels = np.asarray(labels, dtype=int)
    damages = np.asarray(centroid_damages, dtype=float)
    k = int(len(damages))
    if k == 0:
        return {
            "k_ambientes": 0,
            "ambiente_labels": labels,
            "ambiente_damages": np.array([]),
            "ambientes": [],
            "weighted_mean": float("nan"),
            "cluster_to_ambiente": {},
            "silhouette": None,
        }

    mapping, k_amb, sil = _kmeans_damage_groups(damages)
    # Remap every grid cell: original cluster → ambiente
    ambiente_labels = np.vectorize(lambda c: mapping[int(c)], otypes=[int])(labels)

    # Damage of each ambiente = area-weighted mean of member cluster damages
    cell_ha = float(grid_res_m) ** 2 / 10_000.0
    amb_damages = np.zeros(k_amb, dtype=float)
    for a in range(k_amb):
        members = [c for c, am in mapping.items() if am == a]
        areas = np.array([float(np.sum(labels == c)) * cell_ha for c in members], dtype=float)
        dmgs = np.array([float(damages[c]) for c in members], dtype=float)
        total = float(areas.sum())
        amb_damages[a] = float(np.sum(dmgs * areas) / total) if total > 0 else float(np.mean(dmgs))

    ambientes, weighted = cluster_contributions(
        ambiente_labels,
        amb_damages,
        grid_res_m,
        label_prefix="A",
    )

    return {
        "k_ambientes": int(k_amb),
        "ambiente_labels": ambiente_labels,
        "ambiente_damages": amb_damages,
        "ambientes": ambientes,
        "weighted_mean": weighted,
        "cluster_to_ambiente": {int(c): int(a) for c, a in mapping.items()},
        "silhouette": sil,
    }


def _kmeans_damage_groups(damages: np.ndarray) -> tuple[dict[int, int], int, float | None]:
    """Choose k' ≤ k by silhouette on 1D damages; map cluster id → ambiente id."""
    k = len(damages)
    X = damages.reshape(-1, 1).astype(float)
    identity = {i: i for i in range(k)}

    if k <= 1:
        return identity, k, None

    spread = float(np.nanstd(damages))
    # Damages almost identical → single ambiente
    if not np.isfinite(spread) or spread < 1.0:
        return {i: 0 for i in range(k)}, 1, None

    unique = np.unique(np.round(damages, 2))
    if len(unique) == 1:
        return {i: 0 for i in range(k)}, 1, None

    # k=2: silhouette invalid for cand=2; merge if gap < 5 pp else keep both
    if k == 2:
        gap = abs(float(damages[0]) - float(damages[1]))
        if gap < 5.0:
            return {0: 0, 1: 0}, 1, None
        return identity, 2, None

    best_k = k
    best_score = -1.0
    best_labels = np.arange(k)
    # Silhouette needs 2 ≤ k' ≤ n-1 → try 2 .. k-1
    for cand in range(2, k):
        if cand > len(unique):
            break
        model = KMeans(n_clusters=cand, n_init=10, random_state=42)
        lab = model.fit_predict(X)
        n_lab = len(np.unique(lab))
        if n_lab < 2 or n_lab >= k:
            continue
        score = float(silhouette_score(X, lab))
        if score > best_score + 1e-9 or (abs(score - best_score) <= 1e-9 and cand < best_k):
            best_k = cand
            best_score = score
            best_labels = lab

    # If no reduced grouping scored well, keep original k
    if best_k >= k:
        return identity, k, None

    mapping = {i: int(best_labels[i]) for i in range(k)}
    used = sorted(set(mapping.values()))
    remap = {old: new for new, old in enumerate(used)}
    mapping = {c: remap[a] for c, a in mapping.items()}
    return mapping, len(used), best_score


def compare_methods(
    sample_damage: np.ndarray,
    idw_raster: np.ndarray,
    labels: np.ndarray,
    centroid_damages: np.ndarray,
    k: int,
    area_ha: float,
    grid_res_m: float,
) -> tuple[CompareResult, np.ndarray]:
    traditional = float(np.nanmean(sample_damage)) if len(sample_damage) else float("nan")
    finite = idw_raster[np.isfinite(idw_raster)]
    directed = float(np.nanmean(finite)) if len(finite) else float("nan")

    clusters, weighted = cluster_contributions(labels, centroid_damages, grid_res_m)
    reduced = reduce_ambientes_by_damage_similarity(labels, centroid_damages, grid_res_m)
    weighted_amb = float(reduced["weighted_mean"])
    k_amb = int(reduced["k_ambientes"])

    abs_diff = weighted - traditional
    pct_diff = _rel_pct(abs_diff, traditional)

    abs_diff_ambientes = weighted_amb - traditional
    pct_diff_ambientes = _rel_pct(abs_diff_ambientes, traditional)
    abs_diff_k_vs_ambientes = weighted_amb - weighted
    pct_diff_k_vs_ambientes = _rel_pct(abs_diff_k_vs_ambientes, weighted)

    abs_diff_idw = directed - traditional
    pct_diff_idw = _rel_pct(abs_diff_idw, traditional)

    return CompareResult(
        traditional_mean=round(traditional, 4),
        weighted_mean=round(weighted, 4) if np.isfinite(weighted) else 0.0,
        weighted_ambientes_mean=round(weighted_amb, 4) if np.isfinite(weighted_amb) else 0.0,
        directed_mean=round(directed, 4),
        abs_diff=round(abs_diff, 4) if np.isfinite(abs_diff) else 0.0,
        pct_diff=round(pct_diff, 4) if np.isfinite(pct_diff) else 0.0,
        abs_diff_ambientes=round(abs_diff_ambientes, 4) if np.isfinite(abs_diff_ambientes) else 0.0,
        pct_diff_ambientes=round(pct_diff_ambientes, 4) if np.isfinite(pct_diff_ambientes) else 0.0,
        abs_diff_k_vs_ambientes=round(abs_diff_k_vs_ambientes, 4)
        if np.isfinite(abs_diff_k_vs_ambientes)
        else 0.0,
        pct_diff_k_vs_ambientes=round(pct_diff_k_vs_ambientes, 4)
        if np.isfinite(pct_diff_k_vs_ambientes)
        else 0.0,
        abs_diff_idw=round(abs_diff_idw, 4) if np.isfinite(abs_diff_idw) else 0.0,
        pct_diff_idw=round(pct_diff_idw, 4) if np.isfinite(pct_diff_idw) else 0.0,
        n_samples=int(len(sample_damage)),
        k=int(k),
        k_ambientes=k_amb,
        area_ha=round(float(area_ha), 4),
        grid_res_m=float(grid_res_m),
        clusters=clusters,
        ambientes=list(reduced["ambientes"]),
        cluster_to_ambiente=dict(reduced["cluster_to_ambiente"]),
        ambientes_silhouette=reduced["silhouette"],
    ), np.asarray(reduced["ambiente_labels"], dtype=int)


def _rel_pct(diff: float, baseline: float) -> float:
    if baseline == 0:
        return float("nan") if diff != 0 else 0.0
    if not np.isfinite(baseline) or not np.isfinite(diff):
        return float("nan")
    return diff / baseline * 100.0


def summary_stats(values: np.ndarray) -> dict[str, Any]:
    v = values[np.isfinite(values)]
    if len(v) == 0:
        return {"n": 0}
    return {
        "n": int(len(v)),
        "mean": float(np.mean(v)),
        "std": float(np.std(v)),
        "min": float(np.min(v)),
        "max": float(np.max(v)),
        "p5": float(np.percentile(v, 5)),
        "p50": float(np.percentile(v, 50)),
        "p95": float(np.percentile(v, 95)),
    }
