"""
RAG Pipeline — Step 2a: TF-IDF Semantic Chunking
=================================================
Input : chunks.jsonl  (output from step 1 word-count chunker)
Output: chunks_tfidf.jsonl

Strategy
--------
Within each (source, title) group, compute TF-IDF cosine similarity between
every pair of adjacent chunks.  Merge consecutive chunks whose similarity
exceeds `threshold` — provided the merged result stays within `max_words`.

Special handling
----------------
- recipe page_type  : skip similarity merging; merge only same-section parts
  (e.g. "Procedure (part 1)" + "(part 2)") regardless of similarity.
- stub chunks (< MIN_WORDS words) : always merged into their neighbour.

Output schema adds two fields vs the input:
  merged_from  : list[int]   original chunk_ids that were merged
  merge_method : str         "tfidf" | "same_section" | "same_section_split"
                             | "stub" | "post_stub" | "none"
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple
from collections import defaultdict

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ── tunables ──────────────────────────────────────────────────────────────────
INPUT_PATH  = Path("artifacts/chunks/corpus_fixed.jsonl")
OUTPUT_PATH = Path("artifacts/chunks/chunks_tfidf.jsonl")

THRESHOLD   = 0.10   # cosine sim threshold: merge if sim >= this value
                     # recommended range: 0.10 – 0.15
                     # 0.10 → ~1151 chunks, mean 191w  (more aggressive)
                     # 0.15 → ~1244 chunks, mean 177w  (more conservative)

MAX_WORDS   = 260    # hard ceiling for a merged chunk (words)
MIN_WORDS   = 40     # stub threshold: chunks below this are force-merged
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
    """
    Return True if b is a continuation part of the same section as a.
    e.g. sections=["Procedure (part 1)"] and ["Procedure (part 2)"]
    """
    def base_name(sections: List[str]) -> str:
        s = sections[0] if sections else ""
        return re.sub(r"\s*\(part\s*\d+\)\s*$", "", s, flags=re.IGNORECASE).strip()

    return base_name(a["sections"]) == base_name(b["sections"])


def merge_two(a: Dict, b: Dict, method: str) -> Dict:
    merged_text = a["text"].rstrip() + " " + b["text"].lstrip()
    return {
        "source"      : a["source"],
        "page_type"   : a["page_type"],
        "title"       : a["title"],
        "sections"    : a["sections"] + [s for s in b["sections"] if s not in a["sections"]],
        "url"         : a["url"],
        "word_count"  : len(merged_text.split()),
        "text"        : merged_text,
        "merged_from" : a.get("merged_from", [a["chunk_id"]]) + b.get("merged_from", [b["chunk_id"]]),
        "merge_method": method,
        # keep a["chunk_id"] as the representative id (reassigned globally later)
        "chunk_id"    : a["chunk_id"],
    }


# ── per-group processing ──────────────────────────────────────────────────────

def process_recipe_group(group: List[Dict]) -> List[Dict]:
    """
    For recipe chunks: only merge explicit part continuations
    (part 1 + part 2 of the same section).  No similarity merging.
    """
    result: List[Dict] = []
    i = 0
    while i < len(group):
        cur = group[i]
        if i + 1 < len(group) and is_same_section_parts(cur, group[i + 1]) \
                and cur["word_count"] + group[i + 1]["word_count"] <= MAX_WORDS:
            merged = merge_two(cur, group[i + 1], "same_section")
            # keep merging if there is a part 3, 4, ...
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
            result.append(cur)
            i += 1
    return result


def process_group_tfidf(
    group: List[Dict],
    tfidf_matrix,        # sparse matrix, all chunks
) -> List[Dict]:
    """
    Greedy left-to-right merge within a non-recipe group.

    Pass 1 — stub absorption: any chunk < MIN_WORDS is merged into the
             adjacent chunk (prefer next, fall back to previous).
    Pass 2 — TF-IDF similarity merging.
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
                    # try merging forward
                    if i + 1 < len(chunks) and \
                       c["word_count"] + chunks[i+1]["word_count"] <= MAX_WORDS:
                        out.append(merge_two(c, chunks[i+1], "stub"))
                        i += 2
                        changed = True
                        continue
                    # try merging backward
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
        return group

    # ── Pass 2: similarity-based merging ────────────────────────────────────
    # Precompute pairwise similarities between consecutive chunks in this group
    ids = [c["chunk_id"] for c in group]

    def sim(i_idx: int, j_idx: int) -> float:
        ci, cj = ids[i_idx], ids[j_idx]
        if ci >= tfidf_matrix.shape[0] or cj >= tfidf_matrix.shape[0]:
            return 0.0
        return float(cosine_similarity(tfidf_matrix[ci], tfidf_matrix[cj])[0][0])

    result: List[Dict] = []
    i = 0
    while i < len(group):
        cur = group[i]
        cur.setdefault("merged_from", [cur["chunk_id"]])
        cur.setdefault("merge_method", "none")

        # Try to greedily absorb the next chunk
        while i + 1 < len(group):
            nxt = group[i + 1]
            if cur["word_count"] + nxt["word_count"] > MAX_WORDS:
                break
            # recompute sim between current accumulated chunk and next
            # use the *last original id* in cur as proxy
            proxy_id = cur.get("merged_from", [cur["chunk_id"]])[-1]
            next_id  = nxt["chunk_id"]
            if proxy_id < tfidf_matrix.shape[0] and next_id < tfidf_matrix.shape[0]:
                s = float(cosine_similarity(
                    tfidf_matrix[proxy_id], tfidf_matrix[next_id]
                )[0][0])
            else:
                s = 0.0

            if s < THRESHOLD:
                break

            cur = merge_two(cur, nxt, "tfidf")
            i += 1  # consumed nxt

        result.append(cur)
        i += 1

    return result


# ── post-processing ───────────────────────────────────────────────────────────

def post_process_chunks(chunks: List[Dict], min_words: int, max_words: int) -> List[Dict]:
    """
    Two-pass post-processor applied AFTER all per-group merging is done.

    Pass 1 — break oversized same_section chunks:
      A same_section merge that exceeds max_words is split back at the
      sentence boundary nearest to the midpoint.  Each sub-chunk keeps a
      descriptive section label ("… (part N)") and merge_method="same_section_split".

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
                # prefer merging into previous same-title chunk
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
                    })
                    changed = True
                # fall back to next same-title chunk
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
                    })
                    i += 2
                    changed = True
                    continue
                else:
                    out.append(c)   # truly isolated — cannot merge
            else:
                out.append(c)
            i += 1
        fixed = out

    return fixed


# ── main ──────────────────────────────────────────────────────────────────────

def semantic_chunk_tfidf(
    input_path: Path = INPUT_PATH,
    output_path: Path = OUTPUT_PATH,
    threshold: float = THRESHOLD,
    max_words: int = MAX_WORDS,
    min_words: int = MIN_WORDS,
) -> None:
    global MAX_WORDS, MIN_WORDS, THRESHOLD
    MAX_WORDS = max_words
    MIN_WORDS = min_words
    THRESHOLD = threshold

    print(f"Loading chunks from {input_path} …")
    chunks = load_chunks(input_path)
    print(f"  {len(chunks)} input chunks.")

    # Build TF-IDF matrix over ALL chunks (indexed by chunk_id)
    print("Building TF-IDF matrix …")
    texts = [c["text"] for c in chunks]
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=15000,
        ngram_range=(1, 2),       # bigrams capture "stir fry", "soy sauce"
        min_df=2,
        sublinear_tf=True,        # log(1+tf) — helps with long passages
    )
    tfidf_matrix = vectorizer.fit_transform(texts)
    print(f"  Matrix shape: {tfidf_matrix.shape}")

    groups = group_by_title(chunks)
    print(f"  {len(groups)} (source, title) groups.")

    # Process each group
    all_output: List[Dict] = []
    recipe_groups = sum(1 for g in groups.values() if g[0]["page_type"] == "recipe")
    print(f"  {recipe_groups} recipe groups (part-only merging).")

    for group in groups.values():
        if group[0]["page_type"] == "recipe":
            processed = process_recipe_group(group)
        else:
            processed = process_group_tfidf(group, tfidf_matrix)
        all_output.extend(processed)

    # ── Post-processing: fix oversized same_section + absorb remaining stubs ──
    print("Post-processing: splitting oversized chunks and absorbing stubs …")
    before_pp = len(all_output)
    all_output = post_process_chunks(all_output, min_words=min_words, max_words=max_words)
    print(f"  {before_pp} → {len(all_output)} chunks after post-processing.")

    # Reassign sequential chunk_ids
    for new_id, chunk in enumerate(all_output):
        chunk["new_chunk_id"] = new_id

    # Reorder fields
    key_order = ["new_chunk_id", "source", "page_type", "title",
                 "sections", "url", "word_count",
                 "merged_from", "merge_method", "text"]
    all_output = [{k: c[k] for k in key_order if k in c} for c in all_output]

    # Write
    with open(output_path, "w", encoding="utf-8") as fh:
        for chunk in all_output:
            fh.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    # ── stats ────────────────────────────────────────────────────────────────
    import statistics
    wcs = [c["word_count"] for c in all_output]
    methods = defaultdict(int)
    for c in all_output:
        methods[c.get("merge_method", "none")] += 1
    merged_sizes = [len(c["merged_from"]) for c in all_output if len(c.get("merged_from", [])) > 1]

    print(f"\nDone.  {len(all_output)} chunks → {output_path}")
    print(f"  Threshold: {threshold}  |  max_words: {max_words}")
    print(f"  Word count  min={min(wcs)}  max={max(wcs)}"
          f"  mean={statistics.mean(wcs):.0f}"
          f"  median={statistics.median(wcs):.0f}")
    print(f"  Merge method breakdown:")
    for m, cnt in sorted(methods.items(), key=lambda x: -x[1]):
        print(f"    {m:15s}: {cnt:5d} chunks")
    if merged_sizes:
        print(f"  Avg originals per merged chunk: {sum(merged_sizes)/len(merged_sizes):.2f}")

    # sample
    print(f"\nSample output:")
    for c in all_output[:4]:
        preview = c["text"][:100].replace("\n", " ")
        print(f"  [{c['new_chunk_id']}] {c['title']} / {c['sections']} "
              f"({c['word_count']}w, merged_from={c['merged_from']}) — {preview}…")


def run_tfidf_chunking(
    input_path: Path | str = INPUT_PATH,
    output_path: Path | str = OUTPUT_PATH,
    threshold: float = THRESHOLD,
    max_words: int = MAX_WORDS,
    min_words: int = MIN_WORDS,
) -> Path:
    """
    Notebook-friendly wrapper for TF-IDF semantic chunking.

    Args:
        input_path:  Input chunk JSONL path (usually fixed chunks).
        output_path: Output TF-IDF chunk JSONL path.
        threshold:   Cosine similarity threshold for merging.
        max_words:   Hard word ceiling per chunk.
        min_words:   Stub threshold; chunks below this are force-merged.

    Returns:
        Output path (Path) for convenient downstream use.
    """
    in_path = Path(input_path)
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    semantic_chunk_tfidf(
        input_path=in_path,
        output_path=out_path,
        threshold=threshold,
        max_words=max_words,
        min_words=min_words,
    )
    return out_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="TF-IDF semantic chunker")
    parser.add_argument("--input",     default=str(INPUT_PATH))
    parser.add_argument("--output",    default=str(OUTPUT_PATH))
    parser.add_argument("--threshold", type=float, default=THRESHOLD,
                        help="Cosine sim threshold for merging (default 0.10)")
    parser.add_argument("--max-words", type=int, default=MAX_WORDS,
                        help="Hard word ceiling per chunk (default 280)")
    parser.add_argument("--min-words", type=int, default=MIN_WORDS,
                        help="Stub threshold — force-merge below this (default 40)")
    args = parser.parse_args()

    run_tfidf_chunking(
        input_path=Path(args.input),
        output_path=Path(args.output),
        threshold=args.threshold,
        max_words=args.max_words,
        min_words=args.min_words,
    )
