#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------
# Config (edit here directly)
# ----------------------------
ROOT_DIR = Path(__file__).resolve().parents[1]
INPUT_FILE = (
    ROOT_DIR
    / "artifacts"
    / "chunking"
    / "corpus_fixed_size.jsonl"
)
OUTPUT_FILE = (
    ROOT_DIR
    / "artifacts"
    / "chunking"
    / "corpus_tfidf.jsonl"
)
HISTOGRAM_FILE = (
    ROOT_DIR
    / "artifacts"
    / "chunking"
    / "corpus_tfidf_word_count_hist.png"
)

# Semantic subchunk length constraints (in words)
MIN_SUBCHUNK_WORDS = 60
TARGET_SUBCHUNK_WORDS = 80
MAX_SUBCHUNK_WORDS = 120
HIST_BINS = 100

# Adjacent sentence semantic break thresholds
HARD_BREAK_SIM = 0.18
SOFT_BREAK_SIM = 0.30

SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n{2,}")
WORD_RE = re.compile(r"\b[\w'-]+\b")


def normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def split_sentences(text: str) -> List[str]:
    text = text.strip()
    if not text:
        return []
    return [s for s in (normalize_ws(p) for p in SENTENCE_SPLIT_RE.split(text)) if s]


def word_count(text: str) -> int:
    return len(WORD_RE.findall(text))


def adjacent_similarities(sentences: List[str]) -> List[float]:
    if len(sentences) < 2:
        return []
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), lowercase=True)
    mat = vectorizer.fit_transform(sentences)
    sims: List[float] = []
    for i in range(len(sentences) - 1):
        sims.append(float(cosine_similarity(mat[i], mat[i + 1])[0, 0]))
    return sims


def semantic_split_sentences(sentences: List[str]) -> List[str]:
    if not sentences:
        return []
    if len(sentences) == 1:
        return [sentences[0]]

    sims = adjacent_similarities(sentences)
    sent_wc = [word_count(s) for s in sentences]

    groups: List[List[str]] = [[sentences[0]]]
    current_words = sent_wc[0]

    for i in range(1, len(sentences)):
        sim_prev = sims[i - 1] if i - 1 < len(sims) else 1.0
        this_words = sent_wc[i]

        # Force break when current segment is already large.
        if current_words >= MAX_SUBCHUNK_WORDS:
            groups.append([sentences[i]])
            current_words = this_words
            continue

        hard_break = sim_prev < HARD_BREAK_SIM and current_words >= MIN_SUBCHUNK_WORDS
        soft_break = sim_prev < SOFT_BREAK_SIM and current_words >= TARGET_SUBCHUNK_WORDS

        if hard_break or soft_break:
            groups.append([sentences[i]])
            current_words = this_words
        else:
            groups[-1].append(sentences[i])
            current_words += this_words

    # Merge very short tail to previous segment for stability.
    if len(groups) >= 2:
        tail_text = normalize_ws(" ".join(groups[-1]))
        if word_count(tail_text) < MIN_SUBCHUNK_WORDS:
            groups[-2].extend(groups[-1])
            groups.pop()

    return [normalize_ws(" ".join(g)) for g in groups if g]


def load_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_subchunks(rows: List[Dict]) -> List[Dict]:
    out: List[Dict] = []
    global_idx = 1

    for row in rows:
        parent_id = str(row.get("chunk_id", "unknown_parent"))
        text = str(row.get("text", "")).strip()
        sentences = split_sentences(text)
        if not sentences:
            continue

        sub_texts = semantic_split_sentences(sentences)
        for sub_idx, sub_text in enumerate(sub_texts, start=1):
            new_row = dict(row)
            new_row["parent_chunk_id"] = parent_id
            new_row["chunk_id"] = f"tfidf_{global_idx:06d}"
            new_row["subchunk_index"] = sub_idx
            new_row["split_method"] = "tfidf_cosine"
            new_row["text"] = sub_text
            new_row["word_count"] = word_count(sub_text)
            out.append(new_row)
            global_idx += 1

    return out


def save_word_count_histogram(rows: List[Dict]) -> None:
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
    plt.axvline(MIN_SUBCHUNK_WORDS, color="#d62728", linestyle="--", linewidth=1.8, label=f"MIN={MIN_SUBCHUNK_WORDS}")
    plt.axvline(MAX_SUBCHUNK_WORDS, color="#2ca02c", linestyle="--", linewidth=1.8, label=f"MAX={MAX_SUBCHUNK_WORDS}")
    plt.axvspan(MIN_SUBCHUNK_WORDS, MAX_SUBCHUNK_WORDS, color="#2ca02c", alpha=0.08)
    plt.title("TF-IDF Chunk Word Count Distribution")
    plt.xlabel("Word Count")
    plt.ylabel("Frequency")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(HISTOGRAM_FILE, dpi=150)
    plt.close()
    print(f"Histogram: {HISTOGRAM_FILE}")


def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    parent_rows = load_jsonl(INPUT_FILE)
    subchunk_rows = build_subchunks(parent_rows)
    write_jsonl(OUTPUT_FILE, subchunk_rows)

    print(f"Input: {INPUT_FILE}")
    print(f"Output: {OUTPUT_FILE}")
    print(f"Parent chunks: {len(parent_rows)}")
    print(f"Subchunks: {len(subchunk_rows)}")
    if subchunk_rows:
        wc = [int(r["word_count"]) for r in subchunk_rows]
        print(f"Word count min/max: {min(wc)}/{max(wc)}")
    save_word_count_histogram(subchunk_rows)


if __name__ == "__main__":
    main()
