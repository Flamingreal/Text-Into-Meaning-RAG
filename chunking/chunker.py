"""
RAG Pipeline — Step 1: Structural Word-Count Chunking
======================================================
Target model : Qwen2.5-0.5B-Instruct
Strategy     : respect source record boundaries (each record = one section),
               merge tiny sections upward within the same title, then split
               oversized sections at sentence boundaries.
Chunk size   : target ~150 words, hard ceiling 200 words.

Why 150 / 200 words?
  - Qwen2.5-0.5B context window is 32 k tokens, but its retrieval quality
    degrades with noisy, long chunks.  Small models benefit from focused,
    dense chunks.
  - ~150 words ≈ 200-220 tokens (English), leaving comfortable room for
    the question + answer in the prompt.
  - 58 % of existing records are already ≤ 150 words, so most records pass
    through unchanged.  Only very short (<50 w) records are merged, and
    only long ones (>200 w) are split.

Output schema per chunk (JSONL):
  chunk_id   : int, 0-indexed, globally unique
  source     : str
  page_type  : str
  title      : str
  sections   : list[str]   # section name(s) covered by this chunk
  url        : str
  text       : str
  word_count : int
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Any

#  tunables 
INPUT_PATH   = Path("corpus/corpus.jsonl")          # adjust if needed
OUTPUT_PATH  = Path("artifacts/chunks/corpus_fixed.jsonl")
TARGET_WORDS = 150   # soft target: merge below this
MAX_WORDS    = 200   # hard ceiling: split above this
# 


def word_count(text: str) -> int:
    return len(text.split())


def split_into_sentences(text: str) -> List[str]:
    """
    Rough sentence splitter that preserves sentence-ending punctuation.
    Handles common abbreviations well enough for English food-domain text.
    """
    # split on . ! ? followed by whitespace + capital letter
    parts = re.split(r'(?<=[.!?])\s+(?=[A-Z"\'])', text)
    return [p.strip() for p in parts if p.strip()]


def split_text_by_words(text: str, max_words: int) -> List[str]:
    """
    Split a long text into sub-chunks at sentence boundaries,
    each at most `max_words` words.
    Falls back to hard word-count split only when a single sentence
    exceeds the limit.
    """
    sentences = split_into_sentences(text)
    chunks: List[str] = []
    current: List[str] = []
    current_wc = 0

    for sent in sentences:
        swc = word_count(sent)
        if current_wc + swc > max_words and current:
            chunks.append(" ".join(current))
            current, current_wc = [], 0
        # single sentence longer than max → hard split by words
        if swc > max_words:
            words = sent.split()
            for i in range(0, len(words), max_words):
                sub = " ".join(words[i : i + max_words])
                if sub:
                    chunks.append(sub)
            continue
        current.append(sent)
        current_wc += swc

    if current:
        chunks.append(" ".join(current))

    return chunks


def load_records(path: Path) -> List[Dict[str, Any]]:
    records = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def group_by_title(records: List[Dict]) -> Dict[str, List[Dict]]:
    """
    Preserve original order; group by (source, title) key.
    Returns an ordered dict of lists.
    """
    groups: Dict[str, List[Dict]] = {}
    for rec in records:
        key = f"{rec['source']}|||{rec['title']}"
        groups.setdefault(key, []).append(rec)
    return groups


def merge_and_split(records: List[Dict], target: int, max_wds: int) -> List[Dict]:
    """
    Given a list of section-records that share the same (source, title):
      1. Merge consecutive tiny sections (< target words) into one chunk.
      2. Pass through sections that are already within [target, max_wds].
      3. Split sections that exceed max_wds at sentence boundaries.

    Returns a list of proto-chunk dicts (without chunk_id yet).
    """
    result: List[Dict] = []

    # buffer for merging small sections
    buf_text: List[str] = []
    buf_sections: List[str] = []
    buf_wc = 0

    def flush_buffer():
        if buf_text:
            merged = " ".join(buf_text)
            result.append({
                "source"    : records[0]["source"],
                "page_type" : records[0]["page_type"],
                "title"     : records[0]["title"],
                "sections"  : buf_sections[:],
                "url"       : records[0]["url"],
                "text"      : merged,
                "word_count": word_count(merged),
            })
            buf_text.clear()
            buf_sections.clear()

    for rec in records:
        text = rec["text"].strip()
        section = rec.get("section", "")
        wc = word_count(text)

        if wc <= target:
            # small section: accumulate
            if buf_wc + wc > max_wds:
                flush_buffer()
            buf_text.append(text)
            buf_sections.append(section)
            buf_wc += wc

        elif wc <= max_wds:
            # fits fine: flush any pending buffer first, then emit as-is
            flush_buffer()
            result.append({
                "source"    : rec["source"],
                "page_type" : rec["page_type"],
                "title"     : rec["title"],
                "sections"  : [section],
                "url"       : rec["url"],
                "text"      : text,
                "word_count": wc,
            })

        else:
            # oversized: flush buffer, then split at sentence boundaries
            flush_buffer()
            sub_texts = split_text_by_words(text, max_wds)
            for i, sub in enumerate(sub_texts):
                label = f"{section} (part {i+1})" if len(sub_texts) > 1 else section
                result.append({
                    "source"    : rec["source"],
                    "page_type" : rec["page_type"],
                    "title"     : rec["title"],
                    "sections"  : [label],
                    "url"       : rec["url"],
                    "text"      : sub,
                    "word_count": word_count(sub),
                })

    flush_buffer()
    return result


def chunk_corpus(input_path: Path, output_path: Path,
                 target: int = TARGET_WORDS,
                 max_wds: int = MAX_WORDS) -> None:
    print(f"Loading records from {input_path} …")
    records = load_records(input_path)
    print(f"  {len(records)} records loaded.")

    groups = group_by_title(records)
    print(f"  {len(groups)} (source, title) groups found.")

    all_chunks: List[Dict] = []
    for group_records in groups.values():
        proto_chunks = merge_and_split(group_records, target, max_wds)
        all_chunks.extend(proto_chunks)

    # assign global numeric chunk_id
    for idx, chunk in enumerate(all_chunks):
        chunk["chunk_id"] = idx

    # reorder keys for readability
    key_order = ["chunk_id", "source", "page_type", "title",
                 "sections", "url", "word_count", "text"]
    all_chunks = [{k: c[k] for k in key_order} for c in all_chunks]

    # write output
    with open(output_path, "w", encoding="utf-8") as fh:
        for chunk in all_chunks:
            fh.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    #  summary stats 
    wcs = [c["word_count"] for c in all_chunks]
    import statistics
    print(f"\nDone.  {len(all_chunks)} chunks written to {output_path}")
    print(f"  word count  min={min(wcs)}  max={max(wcs)}"
          f"  mean={statistics.mean(wcs):.0f}"
          f"  median={statistics.median(wcs):.0f}")
    over = sum(1 for w in wcs if w > max_wds)
    print(f"  chunks exceeding hard ceiling ({max_wds} w): {over}")
    print(f"\nSample chunks:")
    for c in all_chunks[:3]:
        preview = c["text"][:120].replace("\n", " ")
        print(f"  [{c['chunk_id']}] {c['title']} / {c['sections']} "
              f"({c['word_count']} w) — {preview}…")


def run_fixed_chunking(
    input_path: Path | str = INPUT_PATH,
    output_path: Path | str = OUTPUT_PATH,
    target_words: int = TARGET_WORDS,
    max_words: int = MAX_WORDS,
) -> Path:
    """
    Notebook-friendly wrapper for fixed-size structural chunking.

    Args:
        input_path:   Source corpus JSONL path.
        output_path:  Destination chunk JSONL path.
        target_words: Soft merge target in words.
        max_words:    Hard split ceiling in words.

    Returns:
        Output path (Path) for convenient downstream use.
    """
    in_path = Path(input_path)
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    chunk_corpus(
        input_path=in_path,
        output_path=out_path,
        target=target_words,
        max_wds=max_words,
    )
    return out_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Corpus chunker for RAG pipeline")
    parser.add_argument("--input",  default=str(INPUT_PATH),  help="Input JSONL path")
    parser.add_argument("--output", default=str(OUTPUT_PATH), help="Output JSONL path")
    parser.add_argument("--target", type=int, default=TARGET_WORDS,
                        help="Soft merge target in words (default 150)")
    parser.add_argument("--max",    type=int, default=MAX_WORDS,
                        help="Hard split ceiling in words (default 200)")
    args = parser.parse_args()

    run_fixed_chunking(
        input_path=Path(args.input),
        output_path=Path(args.output),
        target_words=args.target,
        max_words=args.max,
    )
