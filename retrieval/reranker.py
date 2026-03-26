"""
Cross-encoder reranker.

Scores every (query, passage) pair and reorders the candidate list.
Uses the same local-first model resolution as the rest of the pipeline.
"""
from pathlib import Path
from typing import Any, Dict, List, Tuple

DEFAULT_CROSS_ENCODER = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def _resolve_model(
    model_ref: str,
    checkpoints_dir: Path = Path("model_checkpoints"),
) -> str:
    p = Path(model_ref)
    if p.exists() and p.is_dir():
        return model_ref
    basename = p.name
    safe_name = str(p).replace("/", "__")
    for name in (basename, safe_name):
        candidate = checkpoints_dir / name
        if candidate.exists() and candidate.is_dir():
            print(f"[Reranker] Local checkpoint: {candidate.resolve()}")
            return str(candidate)
    print(f"[Reranker] Using HuggingFace: {model_ref}")
    return model_ref


class CrossEncoderReranker:
    """
    Wraps sentence-transformers CrossEncoder for passage reranking.

    The model scores each (query, passage_text) pair; higher score = more
    relevant.  Only the ``text`` field of the meta doc is used as the passage
    (not embed_text) to avoid prefix noise in cross-attention scoring.
    """

    def __init__(
        self,
        model_ref: str = DEFAULT_CROSS_ENCODER,
        max_length: int = 512,
    ) -> None:
        from sentence_transformers import CrossEncoder

        resolved = _resolve_model(model_ref)
        print(f"[Reranker] Loading cross-encoder: {resolved}")
        self.model = CrossEncoder(resolved, max_length=max_length)

    def rerank(
        self,
        query: str,
        candidates: List[Tuple[int, float, Dict[str, Any]]],
        top_k: int = 10,
    ) -> List[Tuple[int, float, Dict[str, Any]]]:
        """
        Score all candidates and return top_k sorted by cross-encoder score.

        Args:
            query:      Raw query string.
            candidates: (doc_id, prev_score, meta) list (output of rrf_fuse).
            top_k:      Number of results to return.

        Returns:
            (doc_id, cross_encoder_score, meta) list, length ≤ top_k.
        """
        if not candidates:
            return []

        pairs = [(query, c[2].get("text", "")) for c in candidates]
        scores = self.model.predict(pairs, show_progress_bar=False)

        ranked = sorted(
            zip(scores, candidates),
            key=lambda x: float(x[0]),
            reverse=True,
        )
        return [
            (doc_id, float(score), meta)
            for score, (doc_id, _prev, meta) in ranked[:top_k]
        ]
