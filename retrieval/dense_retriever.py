"""
Dense retriever: FAISS index + SentenceTransformer query encoder.

Reads a manifest.json produced by embed_chunks.py to locate the index and
meta files automatically.  The query is encoded with the Qwen3-Embedding
query instruction before searching.
"""
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import faiss
import numpy as np

# Qwen3-Embedding query instruction (fallback when prompt_name is unavailable)
_QUERY_INSTRUCTION = (
    "Instruct: Given a web search query, retrieve relevant passages "
    "that answer the query\nQuery: "
)


def _resolve_model(
    model_ref: str,
    checkpoints_dir: Path = Path("model_checkpoints"),
) -> str:
    """
    Same local-first resolution as embed_chunks.py:
      1. model_ref is an existing directory → use as-is.
      2. <checkpoints_dir>/<basename> exists → use it.
      3. <checkpoints_dir>/<org>__<name> exists → use it.
      4. Fall back to model_ref as a HuggingFace ID.
    """
    p = Path(model_ref)
    if p.exists() and p.is_dir():
        return model_ref

    basename = p.name
    safe_name = str(p).replace("/", "__")
    for name in (basename, safe_name):
        candidate = checkpoints_dir / name
        if candidate.exists() and candidate.is_dir():
            print(f"[Dense] Local checkpoint found: {candidate.resolve()}")
            return str(candidate)

    print(f"[Dense] Local model not found, using HuggingFace: {model_ref}")
    return model_ref


class DenseRetriever:
    """
    Loads a FAISS index + meta from a manifest file and encodes queries
    with the same embedding model used to build the index.
    """

    def __init__(self, manifest_path: Path) -> None:
        with open(manifest_path, encoding="utf-8") as fh:
            self.manifest: Dict[str, Any] = json.load(fh)

        self.model_ref: str = self.manifest["model_ref"]
        self.normalize: bool = self.manifest.get("normalize", True)
        self.max_length: int = self.manifest.get("max_length", 512)

        # Resolve paths: if a manifest-relative path doesn't exist from cwd,
        # fall back to resolving relative to the manifest file's directory.
        manifest_dir = manifest_path.parent

        def _resolve(rel: str) -> Path:
            p = Path(rel)
            if p.exists():
                return p
            candidate = manifest_dir / p.name
            if candidate.exists():
                return candidate
            # last resort: resolve relative to manifest_dir assuming the stored
            # path is relative to the project root (two levels up from manifest)
            candidate2 = (manifest_dir / rel).resolve()
            if candidate2.exists():
                return candidate2
            raise FileNotFoundError(
                f"Cannot locate '{rel}' — tried cwd, {candidate}, {candidate2}"
            )

        # Load meta
        meta_path = _resolve(self.manifest["meta_jsonl_path"])
        self.meta: List[Dict[str, Any]] = []
        with open(meta_path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    self.meta.append(json.loads(line))

        # Load FAISS index
        index_path = _resolve(self.manifest["index_path"])
        self.index: faiss.Index = faiss.read_index(str(index_path))
        print(
            f"[Dense] FAISS loaded: {self.index.ntotal} vectors, "
            f"dim={self.manifest['dimension']} from {index_path.name}"
        )

        self._model = None  # lazy load

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            resolved = _resolve_model(self.model_ref)
            print(f"[Dense] Loading query encoder: {resolved}")
            self._model = SentenceTransformer(resolved, trust_remote_code=True)
            self._model.max_seq_length = self.max_length
        return self._model

    def encode_query(self, query: str) -> np.ndarray:
        """Encode a single query string → (1, D) float32 array."""
        try:
            vec = self.model.encode(
                [query],
                prompt_name="retrieval.query",
                normalize_embeddings=self.normalize,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        except (TypeError, KeyError, AttributeError, ValueError):
            vec = self.model.encode(
                [f"{_QUERY_INSTRUCTION}{query}"],
                normalize_embeddings=self.normalize,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        return vec.astype(np.float32)

    def search(
        self, query: str, top_k: int = 100
    ) -> List[Tuple[int, float, Dict[str, Any]]]:
        """
        Returns list of (vector_id, similarity_score, meta_doc) sorted desc.
        """
        query_vec = self.encode_query(query)
        scores, indices = self.index.search(query_vec, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:
                results.append((int(idx), float(score), self.meta[idx]))
        return results
