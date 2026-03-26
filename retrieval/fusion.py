"""
Reciprocal Rank Fusion (RRF).

Standard RRF formula:
    RRF(d) = Σ_i  w_i / (k + rank_i(d))

where k=60 is the smoothing constant that reduces the impact of high rankings.
"""
from typing import Any, Dict, List, Optional, Tuple


def rrf_fuse(
    ranked_lists: List[List[Tuple[int, float, Dict[str, Any]]]],
    k: int = 60,
    weights: Optional[List[float]] = None,
) -> List[Tuple[int, float, Dict[str, Any]]]:
    """
    Fuse multiple ranked lists with RRF.

    Args:
        ranked_lists: Each element is a list of (doc_id, score, meta) sorted
                      by score descending.  doc_ids must be comparable across
                      lists (use vector_id from meta.jsonl).
        k:            RRF smoothing constant (default 60).
        weights:      Per-list multipliers (default: uniform 1.0).

    Returns:
        Merged list of (doc_id, rrf_score, meta) sorted by rrf_score desc.
    """
    if weights is None:
        weights = [1.0] * len(ranked_lists)

    doc_scores: Dict[int, float] = {}
    doc_meta: Dict[int, Dict[str, Any]] = {}

    for ranked, weight in zip(ranked_lists, weights):
        for rank, (doc_id, _score, meta) in enumerate(ranked, start=1):
            doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + weight / (k + rank)
            doc_meta[doc_id] = meta  # last write wins (identical across lists)

    fused = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
    return [(doc_id, score, doc_meta[doc_id]) for doc_id, score in fused]
