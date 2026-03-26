"""
BM25 sparse retriever.

Uses the FAISS meta.jsonl as corpus so that doc_ids (vector_id) are identical
to those returned by DenseRetriever — enabling direct RRF fusion.
"""
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

from rank_bm25 import BM25Okapi


def _tokenize(text: str) -> List[str]:
    """Lowercase, strip punctuation, split on whitespace."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return text.split()


class BM25Retriever:
    """
    Builds a BM25Okapi index over the ``text`` field of every entry in
    *meta_jsonl_path*.  The doc_id for each entry is its ``vector_id``,
    which matches the FAISS vector position used by DenseRetriever.
    """

    def __init__(self, meta_jsonl_path: Path) -> None:
        self.docs: List[Dict[str, Any]] = []
        tokenized_corpus: List[List[str]] = []

        with open(meta_jsonl_path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                doc = json.loads(line)
                self.docs.append(doc)
                tokenized_corpus.append(_tokenize(doc.get("text", "")))

        self.bm25 = BM25Okapi(tokenized_corpus)
        print(f"[BM25] Indexed {len(self.docs)} docs from {meta_jsonl_path.name}")

    def search(
        self, query: str, top_k: int = 100
    ) -> List[Tuple[int, float, Dict[str, Any]]]:
        """
        Returns list of (vector_id, bm25_score, meta_doc) sorted by score desc.
        Only docs with score > 0 are returned.
        """
        tokens = _tokenize(query)
        scores = self.bm25.get_scores(tokens)

        # argsort descending, take top_k
        ranked_idx = scores.argsort()[::-1][:top_k]
        results = []
        for idx in ranked_idx:
            if scores[idx] > 0:
                doc = self.docs[idx]
                results.append((int(doc["vector_id"]), float(scores[idx]), doc))
        return results
