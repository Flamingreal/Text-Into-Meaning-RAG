"""
RAG Pipeline — Step 3: Chunk Embedding with Natural Language Prefix
===================================================================

Strategy: natural_language_prefix
  Prepend a short human-readable context header to each chunk before
  encoding.  The prefix is derived from page_type and aligns the
  embedding distribution with how users phrase retrieval queries.

Prefix templates (by page_type):
  recipe      : "Recipe for {clean_title}. {clean_sections} section."
  ingredient  : "Ingredient guide: {clean_title}."
  blog        : "Blog: {title}."
  overview    : "About {title}. {clean_sections}."
  sub_cuisine : "About {title}. {clean_sections}."
  (fallback)  : "{title}. {clean_sections}."

clean_title    : strip leading "Cookbook:" (Wikibooks entries)
clean_sections : strip "(part N)" suffixes, deduplicate, join with ", "

Output files (all written under --output-dir, default artifacts/faiss/):
  {stem}.nlp_prefix.{index_type}.index        — FAISS binary index
  {stem}.nlp_prefix.{index_type}.meta.jsonl   — per-vector chunk metadata + embed_text
  {stem}.nlp_prefix.{index_type}.manifest.json — run parameters

stem = {input_stem}.{model_safe_name}
  e.g. chunks_embedding.Qwen__Qwen3-Embedding-4B

Usage examples:
  # default: chunks_embedding.jsonl + 4B model
  python embedding/embed_chunks.py

  # different chunk file
  python embedding/embed_chunks.py --input artifacts/chunks/chunks_tfidf.jsonl

  # different model
  python embedding/embed_chunks.py --model model_checkpoints/Qwen__Qwen3-Embedding-0.6B

  # 0.6B model + tfidf chunks, larger batch
  python embedding/embed_chunks.py \\
      --input  artifacts/chunks/chunks_tfidf.jsonl \\
      --model  model_checkpoints/Qwen__Qwen3-Embedding-0.6B \\
      --batch-size 16
"""

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List

import faiss
import numpy as np

# ── defaults ──────────────────────────────────────────────────────────────────
DEFAULT_INPUT      = Path("artifacts/chunks/chunks_embedding.jsonl")
DEFAULT_MODEL      = "Qwen/Qwen3-Embedding-0.6B"
DEFAULT_OUTPUT_DIR = Path("artifacts/faiss")
DEFAULT_INDEX_TYPE = "FlatIP"   # inner product; with L2 normalisation equals cosine similarity
DEFAULT_NORMALIZE  = True
DEFAULT_BATCH_SIZE = 4
DEFAULT_MAX_LENGTH = 512

# Document-side instruction for Qwen3-Embedding.
# sentence-transformers >= 2.7 can pass this via prompt_name; older versions
# fall back to manual prepending (handled in encode_texts).
DOC_INSTRUCTION = "Represent this passage for retrieval:"

# Local directory where downloaded model checkpoints are stored.
LOCAL_CHECKPOINTS_DIR = Path("model_checkpoints")
# ─────────────────────────────────────────────────────────────────────────────


# ── prefix construction ───────────────────────────────────────────────────────

def _strip_cookbook(title: str) -> str:
    """Strip the Wikibooks 'Cookbook:' prefix from a title."""
    return re.sub(r"^Cookbook:\s*", "", title, flags=re.IGNORECASE).strip()


def _clean_sections(sections: List[str]) -> str:
    """
    Remove '(part N)' suffixes, deduplicate, and join with ', '.
    ['Lead (part 1)', 'Lead (part 2)', 'Ingredients'] -> 'Lead, Ingredients'
    """
    seen: Dict[str, bool] = {}
    cleaned: List[str] = []
    for s in sections:
        base = re.sub(r"\s*\(part\s*\d+\)\s*$", "", s, flags=re.IGNORECASE).strip()
        if base and base not in seen:
            seen[base] = True
            cleaned.append(base)
    return ", ".join(cleaned)


def build_prefix(chunk: Dict[str, Any]) -> str:
    """
    Build a natural-language context prefix based on the chunk's page_type.
    The returned string has no trailing newline; build_embed_text inserts a
    blank line between the prefix and the chunk body.
    """
    page_type   = chunk.get("page_type", "").lower()
    title       = chunk.get("title", "").strip()
    sections    = chunk.get("sections", [])
    sec_str     = _clean_sections(sections)
    clean_title = _strip_cookbook(title)

    if page_type == "recipe":
        # e.g. "Recipe for Broccoli Stir Fry. Ingredients, Procedure section."
        if sec_str:
            return f"Recipe for {clean_title}. {sec_str} section."
        return f"Recipe for {clean_title}."

    elif page_type == "ingredient":
        # e.g. "Ingredient guide: Red Bean Paste."
        return f"Ingredient guide: {clean_title}."

    elif page_type == "blog":
        # Section is typically "Lead" — redundant, so omitted.
        # e.g. "Blog: 12. Japan."
        return f"Blog: {title}."

    elif page_type in ("overview", "sub_cuisine"):
        # e.g. "About Chinese cuisine. History."
        if sec_str:
            return f"About {title}. {sec_str}."
        return f"About {title}."

    else:
        # generic fallback
        if sec_str:
            return f"{title}. {sec_str}."
        return f"{title}."


def build_embed_text(chunk: Dict[str, Any]) -> str:
    """
    Compose the final text fed to the embedding model:
      {prefix}

      {chunk body}
    """
    prefix = build_prefix(chunk)
    text   = chunk.get("text", "").strip()
    return f"{prefix}\n\n{text}"


# ── I/O helpers ───────────────────────────────────────────────────────────────

def load_chunks(path: Path) -> List[Dict[str, Any]]:
    chunks = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def safe_model_name(model_ref: str) -> str:
    """
    Convert a model path or HuggingFace ID to a filesystem-safe string.
    'Qwen/Qwen3-Embedding-0.6B'              -> 'Qwen__Qwen3-Embedding-0.6B'
    'model_checkpoints/Qwen3-Embedding-0.6B' -> 'Qwen3-Embedding-0.6B'
    """
    p = Path(model_ref)
    # For a local path the parent directory exists on disk → use basename only.
    # For a HuggingFace "org/name" ID the parent ("org") does not exist → keep both parts.
    if p.parent != Path(".") and p.parent.exists():
        return p.name
    return str(p).replace("/", "__").replace("\\", "__")


# ── model resolution ──────────────────────────────────────────────────────────

def resolve_model_ref(model_ref: str) -> str:
    """
    Resolve *model_ref* to an actual loadable path or HuggingFace model ID.

    Accepts both HuggingFace IDs (``"Qwen/Qwen3-Embedding-0.6B"``) and
    explicit local paths (``"model_checkpoints/Qwen3-Embedding-0.6B"``).

    Resolution order:
      1. If *model_ref* is already an existing local directory → use it as-is.
      2. Derive candidate paths under LOCAL_CHECKPOINTS_DIR and check both:
           <checkpoints>/<basename>        e.g. model_checkpoints/Qwen3-Embedding-0.6B
           <checkpoints>/<org>__<name>     e.g. model_checkpoints/Qwen__Qwen3-Embedding-0.6B
         The first match is used.
      3. Fall back to *model_ref* as a HuggingFace model ID (downloaded on demand).
    """
    # Step 1: explicit local path that already exists
    p = Path(model_ref)
    if p.exists() and p.is_dir():
        print(f"  [model] 本地模型已找到: {p.resolve()}")
        return model_ref

    # Step 2: look inside LOCAL_CHECKPOINTS_DIR
    basename  = p.name                          # e.g. "Qwen3-Embedding-0.6B"
    safe_name = str(p).replace("/", "__")       # e.g. "Qwen__Qwen3-Embedding-0.6B"
    for candidate_name in (basename, safe_name):
        candidate = LOCAL_CHECKPOINTS_DIR / candidate_name
        if candidate.exists() and candidate.is_dir():
            print(f"  [model] 本地 checkpoint 已找到: {candidate.resolve()}")
            return str(candidate)

    # Step 3: use as HuggingFace ID
    print(
        f"  [model] Not found in local '{model_ref}'，"
        f"Pull from HuggingFace: {model_ref}"
    )
    return model_ref


# ── embedding ─────────────────────────────────────────────────────────────────

def encode_texts(
    texts: List[str],
    model_ref: str,
    batch_size: int,
    max_length: int,
    normalize: bool,
) -> np.ndarray:
    """
    Encode a list of strings with SentenceTransformer and return an (N, D)
    float32 array.

    Instruction passing strategy for Qwen3-Embedding:
      1. Try prompt_name="retrieval.passage" (sentence-transformers >= 2.7
         with model prompt config present).
      2. On failure, fall back to prepending DOC_INSTRUCTION manually.
    """
    from sentence_transformers import SentenceTransformer

    model_ref = resolve_model_ref(model_ref)
    print(f"  Loading model: {model_ref} ...")
    model = SentenceTransformer(model_ref, trust_remote_code=True)

    # Set max sequence length on the model rather than passing to encode(),
    # which not all model versions accept as a keyword argument.
    model.max_seq_length = max_length

    encode_kwargs: Dict[str, Any] = dict(
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=normalize,
        convert_to_numpy=True,
    )

    try:
        t0 = time.time()
        embeddings = model.encode(
            texts,
            prompt_name="retrieval.passage",
            **encode_kwargs,
        )
        print(f"  Encoded via prompt_name='retrieval.passage' in {time.time()-t0:.1f}s")
    except (TypeError, KeyError, AttributeError, ValueError):
        # prompt_name not supported — prepend instruction text directly
        print("  prompt_name unsupported; falling back to manual instruction prepend ...")
        instruct_texts = [f"{DOC_INSTRUCTION}\n{t}" for t in texts]
        t0 = time.time()
        embeddings = model.encode(instruct_texts, **encode_kwargs)
        print(f"  Encoded via manual instruction prepend in {time.time()-t0:.1f}s")

    print(f"  Embedding shape: {embeddings.shape}")
    return embeddings.astype(np.float32)


# ── FAISS index ───────────────────────────────────────────────────────────────

def build_faiss_index(embeddings: np.ndarray, index_type: str) -> faiss.Index:
    dim = embeddings.shape[1]
    if index_type == "FlatIP":
        index = faiss.IndexFlatIP(dim)
    elif index_type == "FlatL2":
        index = faiss.IndexFlatL2(dim)
    else:
        raise ValueError(f"Unsupported index_type '{index_type}'. Choose FlatIP or FlatL2.")
    index.add(embeddings)
    return index


# ── main pipeline ─────────────────────────────────────────────────────────────

def run_embedding(
    input_path:  Path  = DEFAULT_INPUT,
    output_dir:  Path  = DEFAULT_OUTPUT_DIR,
    model_ref:   str   = DEFAULT_MODEL,
    index_type:  str   = DEFAULT_INDEX_TYPE,
    normalize:   bool  = DEFAULT_NORMALIZE,
    batch_size:  int   = DEFAULT_BATCH_SIZE,
    max_length:  int   = DEFAULT_MAX_LENGTH,
    dry_run:     bool  = False,
) -> None:
    """
    Full embedding pipeline:
      1. Load chunk JSONL.
      2. Build natural_language_prefix embed_text for every chunk.
      3. Batch-encode into vectors.
      4. Build and persist a FAISS index.
      5. Write meta JSONL and manifest JSON.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. load chunks ────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Input : {input_path}")
    print(f"Model : {model_ref}")
    print(f"Strategy: natural_language_prefix | Index: {index_type}")
    print(f"{'='*60}\n")

    print("[1/4] Loading chunks ...")
    chunks = load_chunks(input_path)
    print(f"  {len(chunks)} chunks loaded.")

    # ── 2. build embed texts ──────────────────────────────────────────────────
    print("\n[2/4] Building embed_text (natural_language_prefix strategy) ...")
    embed_texts = [build_embed_text(c) for c in chunks]

    # show one prefix example per page_type
    prefix_samples: Dict[str, str] = {}
    for c, et in zip(chunks, embed_texts):
        pt = c.get("page_type", "unknown")
        if pt not in prefix_samples:
            prefix_samples[pt] = et.split("\n\n")[0]
    print("  Prefix samples by page_type:")
    for pt, prefix in sorted(prefix_samples.items()):
        print(f"    [{pt:12s}]  {prefix}")

    if dry_run:
        print("\n  --dry-run: printing first 3 embed_texts then exiting.")
        for i in range(min(3, len(embed_texts))):
            print(f"\n  -- chunk {i} --")
            print(embed_texts[i][:300])
        return

    # ── 3. encode ─────────────────────────────────────────────────────────────
    print("\n[3/4] Encoding vectors ...")
    embeddings = encode_texts(embed_texts, model_ref, batch_size, max_length, normalize)

    # ── 4. build and save FAISS ───────────────────────────────────────────────
    print("\n[4/4] Building FAISS index and saving artifacts ...")
    index = build_faiss_index(embeddings, index_type)
    print(f"  Index contains {index.ntotal} vectors of dimension {embeddings.shape[1]}.")

    # derive output file names
    model_safe = safe_model_name(model_ref)
    stem       = f"{input_path.stem}.{model_safe}.nlp_prefix.{index_type}"

    index_path    = output_dir / f"{stem}.index"
    meta_path     = output_dir / f"{stem}.meta.jsonl"
    manifest_path = output_dir / f"{stem}.manifest.json"

    # save FAISS binary index
    faiss.write_index(index, str(index_path))
    print(f"  ✓ FAISS index  -> {index_path}")

    # save meta JSONL — one entry per vector
    _META_KEYS = ("source", "page_type", "title", "sections", "url", "text")
    with open(meta_path, "w", encoding="utf-8") as fh:
        for i, (chunk, embed_text) in enumerate(zip(chunks, embed_texts)):
            meta: Dict[str, Any] = {"vector_id": i}
            # normalise chunk_id regardless of chunker naming convention
            for id_key in ("new_chunk_id", "chunk_id"):
                if id_key in chunk:
                    meta["chunk_id"] = chunk[id_key]
                    break
            meta.update({k: chunk[k] for k in _META_KEYS if k in chunk})
            meta["embed_text"] = embed_text
            fh.write(json.dumps(meta, ensure_ascii=False) + "\n")
    print(f"  ✓ meta JSONL   -> {meta_path}")

    # save manifest
    manifest: Dict[str, Any] = {
        "input_jsonl_path":  str(input_path),
        "model_ref":         model_ref,
        "model_safe":        model_safe,
        "strategy":          "natural_language_prefix",
        "index_type":        index_type,
        "vector_count":      int(index.ntotal),
        "dimension":         int(embeddings.shape[1]),
        "normalize":         normalize,
        "batch_size":        batch_size,
        "max_length":        max_length,
        "doc_instruction":   DOC_INSTRUCTION,
        "index_path":        str(index_path),
        "meta_jsonl_path":   str(meta_path),
    }
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)
    print(f"  ✓ manifest     -> {manifest_path}")

    print(f"\nDone. {index.ntotal} vectors stored in FAISS.\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="RAG Pipeline Step 3: Embed chunks with natural_language_prefix into FAISS",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--input", default=str(DEFAULT_INPUT),
        help="Input chunk JSONL (corpus_fixed / chunks_tfidf / chunks_embedding)",
    )
    p.add_argument(
        "--output-dir", default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for FAISS output files",
    )
    p.add_argument(
        "--model", default=DEFAULT_MODEL,
        help="Embedding model path or HuggingFace name",
    )
    p.add_argument(
        "--index-type", default=DEFAULT_INDEX_TYPE,
        choices=["FlatIP", "FlatL2"],
        help="FAISS index type (FlatIP + normalisation = cosine similarity)",
    )
    p.add_argument(
        "--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
        help="Encoding batch size (reduce if GPU OOM)",
    )
    p.add_argument(
        "--max-length", type=int, default=DEFAULT_MAX_LENGTH,
        help="Maximum number of tokens per input",
    )
    p.add_argument(
        "--no-normalize", action="store_true",
        help="Disable L2 normalisation (enabled by default; required for FlatIP = cosine)",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Print embed_text samples without running the model",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_embedding(
        input_path  = Path(args.input),
        output_dir  = Path(args.output_dir),
        model_ref   = args.model,
        index_type  = args.index_type,
        normalize   = not args.no_normalize,
        batch_size  = args.batch_size,
        max_length  = args.max_length,
        dry_run     = args.dry_run,
    )
