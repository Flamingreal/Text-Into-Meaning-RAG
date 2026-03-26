#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Tuple

# ----------------------------
# Config (edit here directly)
# ----------------------------
ROOT_DIR = Path(__file__).resolve().parents[1]
INPUT_FILES = [
    ROOT_DIR / "corpus" / "corpus.jsonl",
]
OUTPUT_FILE = (
    ROOT_DIR
    / "artifacts"
    / "chunking"
    / "corpus_fixed_size.jsonl"
)
OUTLIER_OUTPUT_FILE = (
    ROOT_DIR
    / "artifacts"
    / "chunking"
    / "corpus_fixed_size_outliers.jsonl"
)
HISTOGRAM_FILE = (
    ROOT_DIR
    / "artifacts"
    / "chunking"
    / "corpus_fixed_size_word_count_hist.png"
)
OVERLAP_WORDS = 20
MIN_WORDS = 100
MAX_WORDS = 180
MERGE_LAST_CHUNK_BELOW = 80


HIST_BINS = 40
OUTLIER_LOW_WORDS = 60
OUTLIER_HIGH_WORDS = 220

# Grouping key for "by title" chunking.
# Same (source, page_type, title, url) records are merged before chunking.
GROUP_KEYS = ("source", "page_type", "title", "url")


SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n{2,}")
WORD_RE = re.compile(r"\b[\w'-]+\b")
CLAUSE_SPLIT_RE = re.compile(r"\s*[;,]\s*")


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def split_sentences(text: str) -> List[str]:
    text = text.strip()
    if not text:
        return []
    return [s for s in (normalize_whitespace(p) for p in SENTENCE_SPLIT_RE.split(text)) if s]


def count_words(text: str) -> int:
    return len(WORD_RE.findall(text))


def build_tail_clause_fragment(sentence: str, max_words: int) -> Tuple[str, int]:
    """
    Build a tail fragment from comma/semicolon clauses within one sentence.
    Returns (fragment, word_count). Empty fragment means cannot truncate.
    """
    if max_words <= 0:
        return "", 0

    clauses = [c.strip() for c in CLAUSE_SPLIT_RE.split(sentence) if c.strip()]
    if len(clauses) < 2:
        return "", 0

    picked: List[str] = []
    total = 0
    for clause in reversed(clauses):
        wc = count_words(clause)
        if not picked and wc > max_words:
            # First clause already too long; hard-trim by words from the tail.
            words = clause.split()
            fragment = " ".join(words[-max_words:]).strip()
            return fragment, count_words(fragment)

        if total + wc > max_words and picked:
            break

        picked.append(clause)
        total += wc
        if total >= max_words:
            break

    if not picked:
        return "", 0

    fragment = normalize_whitespace(", ".join(reversed(picked)))
    # If fragment is effectively the same as full sentence, skip truncation.
    if fragment == normalize_whitespace(sentence):
        return "", 0
    return fragment, count_words(fragment)


def build_sentence_chunks(text: str) -> List[Dict[str, object]]:
    """Fixed-size chunking with overlap that may truncate on comma/semicolon."""
    sentences = split_sentences(text)
    if not sentences:
        return []

    sentence_wc = [count_words(s) for s in sentences]
    chunks: List[Dict[str, object]] = []
    start = 0
    n = len(sentences)
    carry_prefix = ""

    while start < n:
        end = start
        total = 0

        while end < n:
            w = sentence_wc[end]
            if total + w <= MAX_WORDS:
                total += w
                end += 1
                continue

            # Keep sentence intact. If current chunk is still short,
            # allow one more sentence even if it exceeds MAX_WORDS.
            if total < MIN_WORDS:
                total += w
                end += 1
            break

        if end == start:
            end = start + 1

        text_parts: List[str] = []
        if carry_prefix:
            text_parts.append(carry_prefix)
        text_parts.extend(sentences[start:end])
        chunk_text = normalize_whitespace(" ".join(text_parts))
        chunks.append(
            {
                "start": start,
                "end": end,
                "text": chunk_text,
                "word_count": count_words(chunk_text),
            }
        )

        if end >= n:
            break

        # Build overlap from the end of current chunk backwards.
        overlap = 0
        new_start = end
        next_prefix = ""
        i = end - 1
        while i >= start:
            overlap += sentence_wc[i]
            if overlap <= OVERLAP_WORDS:
                new_start = i
                if overlap >= OVERLAP_WORDS:
                    break
            else:
                # Try truncating this overlap sentence at comma/semicolon.
                remaining = OVERLAP_WORDS - (overlap - sentence_wc[i])
                frag, frag_wc = build_tail_clause_fragment(sentences[i], remaining)
                if frag and frag_wc > 0:
                    new_start = i + 1
                    next_prefix = frag
                else:
                    new_start = i
                break
            i -= 1

        # Safety: guarantee forward progress.
        if new_start <= start and not next_prefix:
            new_start = start + 1

        start = new_start
        carry_prefix = next_prefix

    return chunks


def merge_small_last_chunk(chunks: List[Dict[str, object]]) -> List[Dict[str, object]]:
    """Merge a too-small tail chunk into the previous chunk within one section."""
    if len(chunks) < 2:
        return chunks

    last = chunks[-1]
    if int(last.get("word_count", 0)) >= MERGE_LAST_CHUNK_BELOW:
        return chunks

    prev = chunks[-2]
    merged_text = normalize_whitespace(f"{prev['text']} {last['text']}")
    merged_chunk = {
        "start": prev["start"],
        "end": last["end"],
        "text": merged_text,
        "word_count": count_words(merged_text),
    }
    return chunks[:-2] + [merged_chunk]


def merge_small_first_chunk(chunks: List[Dict[str, object]]) -> List[Dict[str, object]]:
    """Merge a too-small first chunk into the next chunk within one section."""
    if len(chunks) < 2:
        return chunks

    first = chunks[0]
    if int(first.get("word_count", 0)) >= MERGE_LAST_CHUNK_BELOW:
        return chunks

    second = chunks[1]
    merged_text = normalize_whitespace(f"{first['text']} {second['text']}")
    merged_chunk = {
        "start": first["start"],
        "end": second["end"],
        "text": merged_text,
        "word_count": count_words(merged_text),
    }
    return [merged_chunk] + chunks[2:]


def group_records_by_title(jsonl_path: Path) -> OrderedDict[Tuple[str, ...], Dict[str, object]]:
    groups: OrderedDict[Tuple[str, ...], Dict[str, object]] = OrderedDict()

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            key = tuple(str(obj.get(k, "")) for k in GROUP_KEYS)
            if key not in groups:
                groups[key] = {
                    "source": obj.get("source", ""),
                    "page_type": obj.get("page_type", ""),
                    "title": obj.get("title", ""),
                    "url": obj.get("url", ""),
                    "sections": [],
                    "section_parts": OrderedDict(),
                }

            section = str(obj.get("section", "")).strip()
            text = str(obj.get("text", "")).strip()
            if section and section not in groups[key]["sections"]:
                groups[key]["sections"].append(section)

            if text:
                if section not in groups[key]["section_parts"]:
                    groups[key]["section_parts"][section] = []
                groups[key]["section_parts"][section].append(text)

    return groups


def build_merged_chunks() -> List[Dict[str, object]]:
    merged_chunks: List[Dict[str, object]] = []
    global_chunk_idx = 1

    for source_file in INPUT_FILES:
        if not source_file.exists():
            raise FileNotFoundError(f"Missing input file: {source_file}")

        groups = group_records_by_title(source_file)
        grouped_doc_idx = 1

        for group in groups.values():
            source = str(group.get("source", source_file.stem)).replace(" ", "_")
            local_idx = 1
            has_chunks = False

            # Chunk each section independently so one chunk never spans two sections.
            for section_name, section_texts in group["section_parts"].items():
                section_text = "\n\n".join(section_texts).strip()
                if not section_text:
                    continue

                chunks = build_sentence_chunks(section_text)
                if not chunks:
                    continue
                chunks = merge_small_first_chunk(chunks)
                chunks = merge_small_last_chunk(chunks)

                has_chunks = True
                for chunk in chunks:
                    original_chunk_id = f"{source}_{grouped_doc_idx:04d}_{local_idx:04d}"
                    merged_chunks.append(
                        {
                            "source": group["source"],
                            "page_type": group["page_type"],
                            "title": group["title"],
                            "url": group["url"],
                            "section": section_name if section_name else "",
                            "chunk_id": f"chunk_{global_chunk_idx:06d}",
                            "chunk_index": local_idx,
                            "text": chunk["text"],
                            "word_count": chunk["word_count"],
                            "original_chunk_id": original_chunk_id,
                            "merged_from": source_file.name,
                        }
                    )
                    global_chunk_idx += 1
                    local_idx += 1

            if has_chunks:
                grouped_doc_idx += 1

    return merged_chunks


def write_jsonl(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def print_stats(rows: List[Dict[str, object]]) -> None:
    if not rows:
        print("No chunks generated.")
        return

    wc = [int(r["word_count"]) for r in rows]
    in_range = sum(1 for n in wc if MIN_WORDS <= n <= MAX_WORDS)
    under = sum(1 for n in wc if n < MIN_WORDS)
    over = sum(1 for n in wc if n > MAX_WORDS)

    print(f"Output: {OUTPUT_FILE}")
    print(f"Chunks: {len(rows)}")
    print(f"Word count min/max: {min(wc)}/{max(wc)}")
    print(f"In range [{MIN_WORDS}, {MAX_WORDS}]: {in_range}")
    print(f"Under min: {under}")
    print(f"Over max: {over}")


def save_word_count_histogram(rows: List[Dict[str, object]]) -> None:
    if not rows:
        return

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("Skip histogram: matplotlib is not installed.")
        return

    wc = [int(r["word_count"]) for r in rows]
    HISTOGRAM_FILE.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.hist(wc, bins=HIST_BINS, color="#4c72b0", edgecolor="#ffffff", alpha=0.9)
    plt.axvline(MIN_WORDS, color="#d62728", linestyle="--", linewidth=1.8, label=f"MIN={MIN_WORDS}")
    plt.axvline(MAX_WORDS, color="#2ca02c", linestyle="--", linewidth=1.8, label=f"MAX={MAX_WORDS}")
    plt.axvspan(MIN_WORDS, MAX_WORDS, color="#2ca02c", alpha=0.08)
    plt.title("Fixed-Size Chunk Word Count Distribution")
    plt.xlabel("Word Count")
    plt.ylabel("Frequency")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(HISTOGRAM_FILE, dpi=150)
    plt.close()
    print(f"Histogram: {HISTOGRAM_FILE}")


def filter_outlier_chunks(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    outliers: List[Dict[str, object]] = []
    for row in rows:
        wc = int(row.get("word_count", 0))
        if wc < OUTLIER_LOW_WORDS or wc > OUTLIER_HIGH_WORDS:
            outliers.append(row)
    return outliers


def main() -> None:
    rows = build_merged_chunks()
    write_jsonl(OUTPUT_FILE, rows)
    outliers = filter_outlier_chunks(rows)
    write_jsonl(OUTLIER_OUTPUT_FILE, outliers)
    print_stats(rows)
    print(
        f"Outliers (<{OUTLIER_LOW_WORDS} or >{OUTLIER_HIGH_WORDS}): "
        f"{len(outliers)} -> {OUTLIER_OUTPUT_FILE}"
    )
    save_word_count_histogram(rows)


if __name__ == "__main__":
    main()
