"""Fusion of sparse (BM25) and dense retrieval results.

Two strategies:

* Reciprocal Rank Fusion (RRF) — robust, scale-free, the modern default.
* Weighted min-max score fusion — score = a*dense + (1-a)*sparse, where the
  report calls for adaptive rather than fixed interpolation. ``dynamic_alpha``
  implements a lightweight per-query version of Dynamic Alpha Tuning
  (Hsu & Tzeng 2025, ref [1]): keyword-/acronym-heavy technical queries pull
  alpha toward BM25; conversational conceptual queries pull it toward dense.
"""
from __future__ import annotations

from backend.app.services.query_processor import ProcessedQuery


def rrf_fuse(
    ranked_lists: list[list[tuple[str, float]]], k: int = 60
) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for lst in ranked_lists:
        for rank, (pid, _) in enumerate(lst):
            scores[pid] = scores.get(pid, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def _minmax(lst: list[tuple[str, float]]) -> dict[str, float]:
    if not lst:
        return {}
    vals = [s for _, s in lst]
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-9:
        return {pid: 1.0 for pid, _ in lst}
    return {pid: (s - lo) / (hi - lo) for pid, s in lst}


def dynamic_alpha(pq: ProcessedQuery, base_alpha: float = 0.5) -> float:
    """Map query technicality in [0,1] to a dense weight.

    technicality 0.0 (conceptual NL question)  -> alpha ~ base+0.25
    technicality 1.0 (exact keyword/acronym)   -> alpha ~ base-0.30
    """
    alpha = base_alpha + 0.25 - 0.55 * pq.technicality
    return float(min(0.9, max(0.1, alpha)))


def weighted_fuse(
    sparse: list[tuple[str, float]],
    dense: list[tuple[str, float]],
    alpha: float,
) -> list[tuple[str, float]]:
    s, d = _minmax(sparse), _minmax(dense)
    all_ids = set(s) | set(d)
    fused = {
        pid: alpha * d.get(pid, 0.0) + (1 - alpha) * s.get(pid, 0.0)
        for pid in all_ids
    }
    return sorted(fused.items(), key=lambda x: x[1], reverse=True)
