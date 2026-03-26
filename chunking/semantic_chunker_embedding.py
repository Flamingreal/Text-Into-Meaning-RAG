"""
RAG Pipeline — Step 2b: Embedding Semantic Chunking
====================================================
Input : chunks.jsonl  (output from step 1 word-count chunker)
Output: chunks_embedding.jsonl

Strategy
--------
Encode every chunk with a lightweight sentence-transformer model.
Within each (source, title) group, merge consecutive chunks whose
embedding cosine similarity exceeds `threshold` — provided the merged
result stays within `max_words`.

Recommended models (all < 130MB, no GPU required for 1590 chunks):
  - BAAI/bge-small-en-v1.5          (33M params, English, best quality here)
  - intfloat/multilingual-e5-small  (118M params, if multilingual needed)
  - sentence-transformers/all-MiniLM-L6-v2  (22M params, fast baseline)

Threshold calibration
---------------------
bge-small-en-v1.5 cosine similarity ranges:
  Same-section continuations ("part 1/2"):  0.88 – 0.96
  Same topic, adjacent sections:            0.72 – 0.85
  Different topic within same title:        0.50 – 0.70
  Completely different titles:              0.20 – 0.50

Recommended threshold: 0.75 – 0.82
  0.75 → more aggressive merging
  0.82 → more conservative (closer to section boundaries)

To calibrate on your data, run with --calibrate flag.  It prints
similarity stats and a sample of merge candidates at different thresholds.

Special handling
----------------
- recipe page_type  : merge only explicit same-section parts, skip sim merging
- stub chunks (< MIN_WORDS) : always absorbed into neighbour before sim pass

Output schema adds vs input:
  merged_from   : list[int]   original chunk_ids merged into this chunk
  merge_method  : str         "embedding" | "same_section" | "same_section_split"
                              | "stub" | "post_stub" | "none"
  avg_sim       : float | None  mean cosine sim of the merge boundary pairs
"""

import json
import re
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import defaultdict

import numpy as np

# ── tunables ──────────────────────────────────────────────────────────────────
INPUT_PATH  = Path("artifacts/chunks/corpus_fixed.jsonl")
OUTPUT_PATH = Path("artifacts/chunks/chunks_embedding.jsonl")

MODEL_NAME  = "BAAI/bge-small-en-v1.5"
# Alternatives:
#   "intfloat/multilingual-e5-small"
#   "sentence-transformers/all-MiniLM-L6-v2"

THRESHOLD   = 0.78   # cosine sim threshold — merge if sim >= this value
MAX_WORDS   = 260    # hard ceiling for a merged chunk (words)
MIN_WORDS   = 40     # stub threshold: force-merge chunks below this
BATCH_SIZE  = 64     # encoding batch size (reduce if OOM)
# ─────────────────────────────────────────────────────────────────────────────


# ── helpers ───────────────────────────────────────────────────────────────────

def load_chunks(path: Path) -> List[Dict[str, Any]]:
    chunks = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def group_by_title(chunks: List[Dict]) -> Dict[str, List[Dict]]:
    groups: Dict[str, List[Dict]] = {}
    for c in chunks:
        key = f"{c['source']}|||{c['title']}"
        groups.setdefault(key, []).append(c)
    return groups


def is_same_section_parts(a: Dict, b: Dict) -> bool:
    def base_name(sections: List[str]) -> str:
        s = sections[0] if sections else ""
        return re.sub(r"\s*\(part\s*\d+\)\s*$", "", s, flags=re.IGNORECASE).strip()
    return base_name(a["sections"]) == base_name(b["sections"])


def merge_two(a: Dict, b: Dict, method: str, sim: Optional[float] = None) -> Dict:
    merged_text = a["text"].rstrip() + " " + b["text"].lstrip()
    prev_sims = a.get("_boundary_sims", [])
    if sim is not None:
        prev_sims = prev_sims + [sim]
    merged = {
        "source"      : a["source"],
        "page_type"   : a["page_type"],
        "title"       : a["title"],
        "sections"    : a["sections"] + [s for s in b["sections"] if s not in a["sections"]],
        "url"         : a["url"],
        "word_count"  : len(merged_text.split()),
        "text"        : merged_text,
        "merged_from" : a.get("merged_from", [a["chunk_id"]]) + b.get("merged_from", [b["chunk_id"]]),
        "merge_method": method,
        "chunk_id"    : a["chunk_id"],
        "_boundary_sims": prev_sims,
    }
    return merged


def encode_chunks(
    chunks: List[Dict],
    model_name: str,
    batch_size: int,
) -> np.ndarray:
    """
    Encode all chunks and return an (N, D) float32 numpy array.
    bge models benefit from a query/passage prefix — we use passage prefix here.
    """
    from sentence_transformers import SentenceTransformer

    print(f"Loading model: {model_name} …")
    model = SentenceTransformer(model_name)

    texts = [c["text"] for c in chunks]

    # bge-small-en-v1.5 recommends a passage prefix for encoding documents
    if "bge" in model_name.lower():
        texts = ["Represent this sentence: " + t for t in texts]
    # multilingual-e5 uses "passage: " prefix
    elif "e5" in model_name.lower():
        texts = ["passage: " + t for t in texts]

    print(f"Encoding {len(texts)} chunks in batches of {batch_size} …")
    t0 = time.time()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,   # L2-norm → dot product = cosine sim
        convert_to_numpy=True,
    )
    print(f"  Encoding done in {time.time()-t0:.1f}s. Shape: {embeddings.shape}")
    return embeddings.astype(np.float32)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity for two L2-normalised vectors (dot product)."""
    return float(np.dot(a, b))


# ── per-group processing ──────────────────────────────────────────────────────

def process_recipe_group(group: List[Dict]) -> List[Dict]:
    """Recipes: only merge explicit part continuations, no sim merging."""
    result: List[Dict] = []
    i = 0
    while i < len(group):
        cur = group[i]
        if i + 1 < len(group) and is_same_section_parts(cur, group[i + 1]) \
                and cur["word_count"] + group[i + 1]["word_count"] <= MAX_WORDS:
            merged = merge_two(cur, group[i + 1], "same_section")
            j = i + 2
            while j < len(group) and is_same_section_parts(merged, group[j]):
                if merged["word_count"] + group[j]["word_count"] > MAX_WORDS:
                    break
                merged = merge_two(merged, group[j], "same_section")
                j += 1
            result.append(merged)
            i = j
        else:
            cur.setdefault("merged_from", [cur["chunk_id"]])
            cur.setdefault("merge_method", "none")
            cur.setdefault("_boundary_sims", [])
            result.append(cur)
            i += 1
    return result


def process_group_embedding(
    group: List[Dict],
    embeddings: np.ndarray,   # shape (total_chunks, D), indexed by chunk_id
) -> List[Dict]:
    """
    Two-pass processing for non-recipe groups:
      Pass 1 — stub absorption (< MIN_WORDS → merge with neighbour)
      Pass 2 — embedding similarity merging
    """

    # ── Pass 1: stub absorption ──────────────────────────────────────────────
    def absorb_stubs(chunks: List[Dict]) -> List[Dict]:
        if len(chunks) <= 1:
            return chunks
        changed = True
        while changed:
            changed = False
            out: List[Dict] = []
            i = 0
            while i < len(chunks):
                c = chunks[i]
                if c["word_count"] < MIN_WORDS:
                    if i + 1 < len(chunks) and \
                       c["word_count"] + chunks[i+1]["word_count"] <= MAX_WORDS:
                        out.append(merge_two(c, chunks[i+1], "stub"))
                        i += 2
                        changed = True
                        continue
                    elif out and out[-1]["word_count"] + c["word_count"] <= MAX_WORDS:
                        out[-1] = merge_two(out[-1], c, "stub")
                        changed = True
                        i += 1
                        continue
                out.append(c)
                i += 1
            chunks = out
        return chunks

    group = absorb_stubs(group)

    if len(group) <= 1:
        for c in group:
            c.setdefault("merged_from", [c["chunk_id"]])
            c.setdefault("merge_method", "none")
            c.setdefault("_boundary_sims", [])
        return group

    # ── Pass 2: embedding similarity merging ─────────────────────────────────
    # For merged chunks, use the *last original id* as the embedding proxy
    # (the end of the accumulation is semantically closest to the next chunk)
    def get_embedding(chunk: Dict) -> Optional[np.ndarray]:
        proxy_id = chunk.get("merged_from", [chunk["chunk_id"]])[-1]
        if proxy_id < embeddings.shape[0]:
            return embeddings[proxy_id]
        return None

    result: List[Dict] = []
    i = 0
    while i < len(group):
        cur = group[i]
        cur.setdefault("merged_from", [cur["chunk_id"]])
        cur.setdefault("merge_method", "none")
        cur.setdefault("_boundary_sims", [])

        while i + 1 < len(group):
            nxt = group[i + 1]
            if cur["word_count"] + nxt["word_count"] > MAX_WORDS:
                break

            emb_cur = get_embedding(cur)
            emb_nxt = embeddings[nxt["chunk_id"]] if nxt["chunk_id"] < embeddings.shape[0] else None

            if emb_cur is None or emb_nxt is None:
                break

            s = cosine_sim(emb_cur, emb_nxt)
            if s < THRESHOLD:
                break

            cur = merge_two(cur, nxt, "embedding", sim=s)
            i += 1

        result.append(cur)
        i += 1

    return result


# ── calibration helper ────────────────────────────────────────────────────────

def calibrate(chunks: List[Dict], embeddings: np.ndarray, n_samples: int = 200) -> None:
    """
    Print similarity stats to help calibrate the threshold.
    Samples adjacent chunk pairs within the same title.
    """
    groups = group_by_title(chunks)
    within_sims: List[float] = []

    for group in groups.values():
        ids = [c["chunk_id"] for c in group]
        for a_id, b_id in zip(ids, ids[1:]):
            if a_id < embeddings.shape[0] and b_id < embeddings.shape[0]:
                within_sims.append(cosine_sim(embeddings[a_id], embeddings[b_id]))
            if len(within_sims) >= n_samples * 10:
                break

    arr = np.array(within_sims)
    print("\n=== Calibration: within-title adjacent cosine similarity ===")
    print(f"  n={len(arr)}")
    for p in [10, 25, 50, 75, 90, 95]:
        print(f"  p{p:2d} = {np.percentile(arr, p):.3f}")

    print("\nMerge candidates at different thresholds:")
    for th in [0.70, 0.75, 0.78, 0.82, 0.85]:
        count = (arr >= th).sum()
        print(f"  threshold={th:.2f}: {count:4d}/{len(arr)} pairs would merge"
              f"  ({100*count/len(arr):.1f}%)")

    # show a few boundary examples
    sorted_pairs = sorted(enumerate(within_sims), key=lambda x: abs(x[1] - THRESHOLD))
    print(f"\nSample pairs near threshold={THRESHOLD}:")
    shown = 0
    all_groups = [g for g in groups.values() if len(g) >= 2]
    pair_lookup: Dict[int, Dict] = {}
    for g in all_groups:
        for c in g:
            pair_lookup[c["chunk_id"]] = c
    for g in all_groups:
        ids = [c["chunk_id"] for c in g]
        for a_id, b_id in zip(ids, ids[1:]):
            if a_id < embeddings.shape[0] and b_id < embeddings.shape[0]:
                s = cosine_sim(embeddings[a_id], embeddings[b_id])
                if abs(s - THRESHOLD) < 0.03:
                    ca, cb = pair_lookup[a_id], pair_lookup[b_id]
                    print(f"\n  sim={s:.3f}  [{ca['title']}]")
                    print(f"  A: {ca['sections']} — {ca['text'][:80]}…")
                    print(f"  B: {cb['sections']} — {cb['text'][:80]}…")
                    shown += 1
                    if shown >= 4:
                        return


# ── post-processing ───────────────────────────────────────────────────────────

def post_process_chunks(chunks: List[Dict], min_words: int, max_words: int) -> List[Dict]:
    """
    Two-pass post-processor applied AFTER all per-group merging is done.

    Pass 1 — break oversized same_section chunks:
      A same_section merge that exceeds max_words is split back at the
      sentence boundary nearest to the midpoint.  Each sub-chunk keeps a
      descriptive section label and merge_method="same_section_split".

    Pass 2 — global stub absorption:
      Any chunk still under min_words is merged into its nearest same-title
      neighbour (prefer previous, then next), regardless of section name.
      merge_method becomes "post_stub".
    """

    # ── Pass 1: re-split oversized same_section chunks ───────────────────────
    fixed: List[Dict] = []
    for c in chunks:
        if c.get("merge_method") == "same_section" and c["word_count"] > max_words:
            sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z"\'])', c["text"])
            target = c["word_count"] // 2
            accumulated, buf, parts = 0, [], []
            for sent in sentences:
                sw = len(sent.split())
                if accumulated + sw > target and buf:
                    parts.append(" ".join(buf))
                    buf, accumulated = [sent], sw
                else:
                    buf.append(sent)
                    accumulated += sw
            if buf:
                parts.append(" ".join(buf))

            src_ids = c.get("merged_from", [c.get("chunk_id", 0)])
            base_section = re.sub(r"\s*\(part\s*\d+\)\s*$", "",
                                  c["sections"][0], flags=re.IGNORECASE).strip()
            for i, part_text in enumerate(parts):
                label = f"{base_section} (part {i+1})" if len(parts) > 1 else base_section
                fixed.append({
                    **c,
                    "text"        : part_text,
                    "word_count"  : len(part_text.split()),
                    "sections"    : [label],
                    "merged_from" : [src_ids[min(i, len(src_ids) - 1)]],
                    "merge_method": "same_section_split",
                    "_boundary_sims": [],
                })
        else:
            fixed.append(c)

    # ── Pass 2: global stub absorption ──────────────────────────────────────
    changed = True
    while changed:
        changed = False
        out: List[Dict] = []
        i = 0
        while i < len(fixed):
            c = fixed[i]
            if c["word_count"] < min_words:
                if out and out[-1]["title"] == c["title"] and \
                   out[-1]["word_count"] + c["word_count"] <= max_words:
                    prev = out.pop()
                    merged_text = prev["text"].rstrip() + " " + c["text"].lstrip()
                    out.append({
                        **prev,
                        "text"        : merged_text,
                        "word_count"  : len(merged_text.split()),
                        "sections"    : prev["sections"] + [s for s in c["sections"]
                                                            if s not in prev["sections"]],
                        "merged_from" : prev.get("merged_from", []) + c.get("merged_from", []),
                        "merge_method": "post_stub",
                        "_boundary_sims": prev.get("_boundary_sims", []),
                    })
                    changed = True
                elif (i + 1 < len(fixed)
                      and fixed[i + 1]["title"] == c["title"]
                      and c["word_count"] + fixed[i + 1]["word_count"] <= max_words):
                    nxt = fixed[i + 1]
                    merged_text = c["text"].rstrip() + " " + nxt["text"].lstrip()
                    out.append({
                        **c,
                        "text"        : merged_text,
                        "word_count"  : len(merged_text.split()),
                        "sections"    : c["sections"] + [s for s in nxt["sections"]
                                                         if s not in c["sections"]],
                        "merged_from" : c.get("merged_from", []) + nxt.get("merged_from", []),
                        "merge_method": "post_stub",
                        "_boundary_sims": c.get("_boundary_sims", []),
                    })
                    i += 2
                    changed = True
                    continue
                else:
                    out.append(c)
            else:
                out.append(c)
            i += 1
        fixed = out

    return fixed


# ── main ──────────────────────────────────────────────────────────────────────

def semantic_chunk_embedding(
    input_path  : Path  = INPUT_PATH,
    output_path : Path  = OUTPUT_PATH,
    model_name  : str   = MODEL_NAME,
    threshold   : float = THRESHOLD,
    max_words   : int   = MAX_WORDS,
    min_words   : int   = MIN_WORDS,
    batch_size  : int   = BATCH_SIZE,
    run_calibrate: bool = False,
) -> None:
    global MAX_WORDS, MIN_WORDS, THRESHOLD
    MAX_WORDS = max_words
    MIN_WORDS = min_words
    THRESHOLD = threshold

    print(f"Loading chunks from {input_path} …")
    chunks = load_chunks(input_path)
    print(f"  {len(chunks)} input chunks.")

    # Encode
    embeddings = encode_chunks(chunks, model_name, batch_size)

    if run_calibrate:
        calibrate(chunks, embeddings)
        return

    groups = group_by_title(chunks)
    print(f"\nProcessing {len(groups)} title groups …")

    all_output: List[Dict] = []
    for group in groups.values():
        if group[0]["page_type"] == "recipe":
            processed = process_recipe_group(group)
        else:
            processed = process_group_embedding(group, embeddings)
        all_output.extend(processed)

    # ── Post-processing: fix oversized same_section + absorb remaining stubs ──
    print("Post-processing: splitting oversized chunks and absorbing stubs …")
    before_pp = len(all_output)
    all_output = post_process_chunks(all_output, min_words=min_words, max_words=max_words)
    print(f"  {before_pp} → {len(all_output)} chunks after post-processing.")

    # Finalise: assign new sequential ids, compute avg_sim, clean internal fields
    for new_id, chunk in enumerate(all_output):
        chunk["new_chunk_id"] = new_id
        sims = chunk.pop("_boundary_sims", [])
        chunk["avg_sim"] = round(float(np.mean(sims)), 4) if sims else None

    key_order = ["new_chunk_id", "source", "page_type", "title",
                 "sections", "url", "word_count",
                 "merged_from", "merge_method", "avg_sim", "text"]
    all_output = [{k: c[k] for k in key_order if k in c} for c in all_output]

    with open(output_path, "w", encoding="utf-8") as fh:
        for chunk in all_output:
            fh.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    # ── stats ────────────────────────────────────────────────────────────────
    import statistics
    wcs = [c["word_count"] for c in all_output]
    methods = defaultdict(int)
    for c in all_output:
        methods[c.get("merge_method", "none")] += 1
    merged_sims = [c["avg_sim"] for c in all_output if c.get("avg_sim") is not None]

    print(f"\nDone.  {len(all_output)} chunks → {output_path}")
    print(f"  Model: {model_name}  |  Threshold: {threshold}  |  max_words: {max_words}")
    print(f"  Word count  min={min(wcs)}  max={max(wcs)}"
          f"  mean={statistics.mean(wcs):.0f}"
          f"  median={statistics.median(wcs):.0f}")
    print(f"  Merge method breakdown:")
    for m, cnt in sorted(methods.items(), key=lambda x: -x[1]):
        print(f"    {m:15s}: {cnt:5d} chunks")
    if merged_sims:
        print(f"  Avg boundary sim of merged chunks: {np.mean(merged_sims):.3f}")

    print(f"\nSample output:")
    for c in all_output[:4]:
        preview = c["text"][:100].replace("\n", " ")
        sim_str = f"avg_sim={c['avg_sim']:.3f}" if c["avg_sim"] else "no merge"
        print(f"  [{c['new_chunk_id']}] {c['title']} / {c['sections']} "
              f"({c['word_count']}w, {sim_str}) — {preview}…")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Embedding semantic chunker")
    parser.add_argument("--input",      default=str(INPUT_PATH))
    parser.add_argument("--output",     default=str(OUTPUT_PATH))
    parser.add_argument("--model",      default=MODEL_NAME,
                        help="HuggingFace model name or local path")
    parser.add_argument("--threshold",  type=float, default=THRESHOLD,
                        help="Cosine sim threshold (default 0.78)")
    parser.add_argument("--max-words",  type=int, default=MAX_WORDS,
                        help="Hard word ceiling per chunk (default 260)")
    parser.add_argument("--min-words",  type=int, default=MIN_WORDS,
                        help="Stub threshold (default 40)")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--calibrate",  action="store_true",
                        help="Print similarity stats to help calibrate threshold, then exit")
    args = parser.parse_args()

    semantic_chunk_embedding(
        input_path   = Path(args.input),
        output_path  = Path(args.output),
        model_name   = args.model,
        threshold    = args.threshold,
        max_words    = args.max_words,
        min_words    = args.min_words,
        batch_size   = args.batch_size,
        run_calibrate= args.calibrate,
    )
