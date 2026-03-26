"""
Generation Quality Evaluation
==============================

Two metrics:

Token F1 Score
--------------
  Same normalisation + token overlap as SQuAD open-domain QA evaluation.
  Strips punctuation, lowercases, removes articles, then computes
  precision/recall/F1 over unigrams.

BERTScore
---------
  Contextual embedding similarity between predicted and reference text
  using the ``bert_score`` library.  Returns Precision, Recall, F1.
  Default model: ``roberta-large`` (best quality).
  Use a smaller model (e.g. ``distilbert-base-uncased``) for faster CPU eval.
"""

import json
import re
import string
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ── Token F1 ──────────────────────────────────────────────────────────────────

def _normalize(text: str) -> List[str]:
    """Lowercase, strip articles + punctuation, tokenize on whitespace."""
    text = text.lower()
    # remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # remove articles
    tokens = [t for t in text.split() if t not in {"a", "an", "the"}]
    return tokens


def token_f1(prediction: str, reference: str) -> Dict[str, float]:
    """
    Compute token-level Precision, Recall, F1 between prediction and reference.

    Returns dict with keys ``precision``, ``recall``, ``f1``.
    Returns 0.0 for all scores if either string is empty after normalisation.
    """
    pred_tokens = _normalize(prediction)
    ref_tokens  = _normalize(reference)

    if not pred_tokens or not ref_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    pred_set = {}
    for t in pred_tokens:
        pred_set[t] = pred_set.get(t, 0) + 1

    ref_set = {}
    for t in ref_tokens:
        ref_set[t] = ref_set.get(t, 0) + 1

    common = sum(min(pred_set.get(t, 0), ref_set.get(t, 0)) for t in ref_set)

    if common == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    precision = common / len(pred_tokens)
    recall    = common / len(ref_tokens)
    f1        = 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


# ── BERTScore ─────────────────────────────────────────────────────────────────

def bert_score_batch(
    predictions: List[str],
    references:  List[str],
    model_type:  str = "roberta-large",
    device:      Optional[str] = None,
    verbose:     bool = False,
) -> List[Dict[str, float]]:
    """
    Compute BERTScore for a batch of (prediction, reference) pairs.

    Args:
        predictions: List of generated response strings.
        references:  List of ground-truth answer strings.
        model_type:  BERT variant used for scoring.
                     ``roberta-large``        – highest quality (slower on CPU).
                     ``distilbert-base-uncased`` – faster on CPU.
        device:      ``"cuda"``, ``"cpu"``, or None (auto-detect).
        verbose:     Show progress bar.

    Returns:
        List of dicts, each with ``precision``, ``recall``, ``f1`` (floats 0–1).
    """
    import torch
    from bert_score import score as _bs_score

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    P, R, F = _bs_score(
        predictions,
        references,
        model_type=model_type,
        device=device,
        verbose=verbose,
    )
    return [
        {"precision": float(p), "recall": float(r), "f1": float(f)}
        for p, r, f in zip(P, R, F)
    ]


# ── Dataset alignment ─────────────────────────────────────────────────────────

def align_outputs_to_benchmark(
    output_payload: Dict[str, Any],
    benchmark: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Match each item in *output_payload["results"]* to its benchmark entry by
    ``query_id`` (int/str) or exact ``query`` text.

    Returns a list of dicts:
      {query_id, query, prediction, reference}

    Items with no benchmark match are silently skipped.
    """
    # build lookup: query_id → answer, query_text → answer
    by_id: Dict[Any, str]  = {str(b["query_id"]): b["answer"] for b in benchmark}
    by_q:  Dict[str, str]  = {b["query"]: b["answer"]         for b in benchmark}

    aligned = []
    for item in output_payload.get("results", []):
        ref = by_id.get(str(item.get("query_id"))) or by_q.get(item.get("query", ""))
        if ref is None:
            continue
        aligned.append(
            {
                "query_id":   item.get("query_id"),
                "query":      item.get("query", ""),
                "prediction": item.get("response", ""),
                "reference":  ref,
            }
        )
    return aligned


# ── Full evaluation ───────────────────────────────────────────────────────────

def evaluate_generation(
    output_payload: Dict[str, Any],
    benchmark: List[Dict[str, Any]],
    bertscore_model: str = "roberta-large",
    device: Optional[str] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Compute Token F1 and BERTScore for all matched (prediction, reference) pairs.

    Returns
    -------
    dict with:
      ``per_item``   – list of per-query scores
      ``token_f1``   – aggregate mean {precision, recall, f1}
      ``bert_score`` – aggregate mean {precision, recall, f1}
      ``n``          – number of evaluated items
    """
    aligned = align_outputs_to_benchmark(output_payload, benchmark)
    if not aligned:
        raise ValueError("No matching items found between output and benchmark.")

    predictions = [a["prediction"] for a in aligned]
    references  = [a["reference"]  for a in aligned]

    # Token F1
    tf1_scores = [token_f1(p, r) for p, r in zip(predictions, references)]

    # BERTScore
    print(f"[BERTScore] Scoring {len(predictions)} pairs with {bertscore_model} ...")
    bs_scores = bert_score_batch(
        predictions, references,
        model_type=bertscore_model,
        device=device,
        verbose=verbose,
    )

    # merge per-item
    per_item = []
    for a, tf1, bs in zip(aligned, tf1_scores, bs_scores):
        per_item.append({**a, "token_f1": tf1, "bert_score": bs})

    n = len(per_item)

    def _mean(scores, key):
        return sum(s[key] for s in scores) / n

    return {
        "per_item":   per_item,
        "token_f1":   {
            "precision": _mean(tf1_scores, "precision"),
            "recall":    _mean(tf1_scores, "recall"),
            "f1":        _mean(tf1_scores, "f1"),
        },
        "bert_score": {
            "precision": _mean(bs_scores, "precision"),
            "recall":    _mean(bs_scores, "recall"),
            "f1":        _mean(bs_scores, "f1"),
        },
        "n": n,
    }


# ── Display ───────────────────────────────────────────────────────────────────

def print_gen_metrics(results: Dict[str, Any]) -> None:
    n = results["n"]
    tf1 = results["token_f1"]
    bs  = results["bert_score"]

    print(f"\n{'='*52}")
    print(f"  Generation Evaluation  (n={n})")
    print(f"{'='*52}")
    print(f"  {'Metric':<25} {'Precision':>9} {'Recall':>9} {'F1':>9}")
    print(f"  {'-'*48}")
    print(f"  {'Token F1':<25} {tf1['precision']:>9.4f} {tf1['recall']:>9.4f} {tf1['f1']:>9.4f}")
    print(f"  {'BERTScore':<25} {bs['precision']:>9.4f} {bs['recall']:>9.4f} {bs['f1']:>9.4f}")
    print(f"{'='*52}")
