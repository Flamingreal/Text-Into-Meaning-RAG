"""
Retrieval evaluation metrics.

Ground-truth matching strategy
--------------------------------
benchmark.json stores ``chunk_ids.embedding`` as an integer equal to the
``vector_id`` / ``chunk_id`` in meta.jsonl.  Primary matching is therefore
exact integer ID comparison:

    hit = (retrieved["vector_id"] == gt_chunk_id)

Title+section matching is provided as a secondary utility for debugging.
"""
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional


# ── helpers ───────────────────────────────────────────────────────────────────

def _strip_part(section: str) -> str:
    return re.sub(r"\s*\(part\s*\d+\)\s*$", "", section, flags=re.IGNORECASE).strip()


def _is_hit_by_id(chunk: Dict[str, Any], gt_chunk_id: int) -> bool:
    """Primary: match by integer vector_id (== chunk_ids.embedding)."""
    return int(chunk.get("vector_id", -1)) == gt_chunk_id


def _is_hit_by_title_section(chunk: Dict[str, Any], gt_title: str, gt_section: str) -> bool:
    """Fallback: match by title + cleaned section name."""
    if chunk.get("title", "") != gt_title:
        return False
    cleaned = {_strip_part(s) for s in chunk.get("sections", [])}
    return gt_section in cleaned


# ── benchmark loading ─────────────────────────────────────────────────────────

def load_benchmark(benchmark_path: Path) -> List[Dict[str, Any]]:
    """
    Load a benchmark JSON file into a normalised list of evaluation items.

    Handles two field-name schemas automatically:
      New schema : query_id, query, answer, title, section, chunk_ids.embedding (int)
      Old schema : id,       question, answer, title, section, chunk_ids.embedding (str "emb_XXX")

    chunk_ids.embedding is always returned as an int (``gt_chunk_id``).
    """
    with open(benchmark_path, encoding="utf-8") as fh:
        raw = json.load(fh)

    items = []
    for entry in raw:
        # ── query id ──────────────────────────────────────────────────────────
        query_id = entry.get("query_id") or entry.get("id")

        # ── query text ────────────────────────────────────────────────────────
        query = entry.get("query") or entry.get("question", "")

        # ── ground-truth chunk id ─────────────────────────────────────────────
        emb_raw = entry["chunk_ids"]["embedding"]
        if isinstance(emb_raw, int):
            gt_chunk_id = emb_raw
        else:
            # e.g. "emb_000001" or "emb_new_000002" → trailing integer
            m = re.search(r"\d+$", str(emb_raw))
            gt_chunk_id = int(m.group()) if m else -1

        items.append(
            {
                "query_id":    query_id,
                "query":       query,
                "answer":      entry.get("answer", ""),
                "title":       entry.get("title", ""),
                "section":     entry.get("section", ""),
                "gt_chunk_id": gt_chunk_id,
            }
        )
    return items


# ── per-query metrics ─────────────────────────────────────────────────────────

def recall_at_k(retrieved: List[Dict[str, Any]], gt_chunk_id: int, k: int) -> float:
    return 1.0 if any(_is_hit_by_id(c, gt_chunk_id) for c in retrieved[:k]) else 0.0


def reciprocal_rank(retrieved: List[Dict[str, Any]], gt_chunk_id: int) -> float:
    for rank, chunk in enumerate(retrieved, start=1):
        if _is_hit_by_id(chunk, gt_chunk_id):
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: List[Dict[str, Any]], gt_chunk_id: int, k: int) -> float:
    for rank, chunk in enumerate(retrieved[:k], start=1):
        if _is_hit_by_id(chunk, gt_chunk_id):
            return 1.0 / math.log2(rank + 1)
    return 0.0


# ── full evaluation ───────────────────────────────────────────────────────────

def evaluate(
    pipeline,
    benchmark: List[Dict[str, Any]],
    ks: List[int] = [1, 3, 5, 10],
    verbose: bool = False,
) -> Dict[str, float]:
    """
    Run the pipeline over every benchmark query and compute aggregate metrics.

    Args:
        pipeline:  Object with ``retrieve(query: str) -> List[Dict]``.
                   Each returned dict must contain ``vector_id`` (int).
        benchmark: Output of ``load_benchmark()``.
        ks:        Cut-offs for Recall@k and NDCG@k.
        verbose:   Print per-query hit/miss status when True.

    Returns:
        Dict with keys "MRR", "Recall@k", "NDCG@k" for each k in ks.
    """
    rr_list: List[float] = []
    recall_acc: Dict[int, List[float]] = {k: [] for k in ks}
    ndcg_acc:   Dict[int, List[float]] = {k: [] for k in ks}

    eval_top_k = max(ks)   # retrieve enough results to cover every cut-off

    for item in benchmark:
        hits = pipeline.retrieve(item["query"], top_k=eval_top_k)
        gt_id = item["gt_chunk_id"]

        rr = reciprocal_rank(hits, gt_id)
        rr_list.append(rr)

        for k in ks:
            recall_acc[k].append(recall_at_k(hits, gt_id, k))
            ndcg_acc[k].append(ndcg_at_k(hits, gt_id, k))

        if verbose:
            hit_rank = next(
                (r for r, h in enumerate(hits, 1) if _is_hit_by_id(h, gt_id)),
                None,
            )
            status = f"hit@{hit_rank}" if hit_rank else "MISS"
            print(f"  [{status:>7}]  gt={gt_id:4d}  {item['query'][:65]}")

    n = len(benchmark)
    metrics: Dict[str, float] = {"MRR": sum(rr_list) / n}
    for k in ks:
        metrics[f"Recall@{k}"] = sum(recall_acc[k]) / n
        metrics[f"NDCG@{k}"]   = sum(ndcg_acc[k])   / n

    return metrics


# ── display ───────────────────────────────────────────────────────────────────

def print_metrics(metrics: Dict[str, float]) -> None:
    print("\n" + "=" * 40)
    print(f"  {'Metric':<15}  {'Score':>8}")
    print("-" * 40)
    for name, val in metrics.items():
        print(f"  {name:<15}  {val:>8.4f}")
    print("=" * 40)
