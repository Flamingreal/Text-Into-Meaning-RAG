#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from math import sqrt
from pathlib import Path
from typing import Any, Dict, List, Tuple

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# ----------------------------
# Config (edit here directly)
# ----------------------------
ROOT_DIR = Path(__file__).resolve().parents[1]
INPUT_FILE = ROOT_DIR / "corpus" / "corpus.json"
OUTPUT_FILE = ROOT_DIR / "artifacts" / "chunking" / "hybrid" / "corpus_hybrid_chunks.jsonl"
HISTOGRAM_FILE = ROOT_DIR / "artifacts" / "chunking" / "hybrid" / "corpus_hybrid_word_count_hist.png"

# Two-stage trigger:
# only long records enter fixed-size coarse split.
COARSE_TRIGGER_WORDS = 700

# Coarse split params
COARSE_MAX_WORDS = 1000
COARSE_OVERLAP_WORDS = 120

# Semantic split params
SEM_MIN_WORDS = 120
SEM_TARGET_WORDS = 260
SEM_MAX_WORDS = 420
SEM_ABSORB_BELOW = 80
FINAL_FRAGMENT_MERGE_BELOW = 30

# Final rebalance target range
FINAL_MIN_WORDS = 120
FINAL_TARGET_WORDS = 140
FINAL_MAX_WORDS = 160
STRICT_WINDOW_REBALANCE = True
STRICT_MAX_ITERATIONS = 3
FINAL_TITLE_PACKING = True
POST_PACK_STRICT_REBALANCE = True

# Keep these recipe sections unsplit unless they are very long.
RECIPE_KEEP_SECTIONS = {"ingredients", "procedure"}
RECIPE_FORCE_SPLIT_WORDS = 260
HIST_BINS = 50

SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
WORD_RE = re.compile(r"\b[\w'-]+\b")


def normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def count_words(text: str) -> int:
    return len(WORD_RE.findall(text))


def split_paragraphs(text: str) -> List[str]:
    parts = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    if parts:
        return parts
    one_line = text.strip()
    return [one_line] if one_line else []


def split_sentences(text: str) -> List[str]:
    text = normalize_ws(text)
    if not text:
        return []
    return [s.strip() for s in SENTENCE_SPLIT_RE.split(text) if s.strip()]


def parse_records(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected list JSON in {path}")
    return [x for x in data if isinstance(x, dict)]


def make_parent_doc_id(record: Dict[str, Any], idx: int) -> str:
    existing = str(record.get("doc_id", "")).strip()
    if existing:
        return existing
    source = str(record.get("source", "unknown")).strip() or "unknown"
    title = str(record.get("title", "")).strip()
    section = str(record.get("section", "")).strip()
    raw = f"{source}|{title}|{section}|{idx}"
    digest = hashlib.md5(raw.encode("utf-8")).hexdigest()[:12]
    source_slug = re.sub(r"[^a-zA-Z0-9]+", "_", source).strip("_").lower() or "unknown"
    return f"{source_slug}_auto_{digest}"


def fixed_size_split(text: str, max_words: int, overlap_words: int) -> List[str]:
    sentences = split_sentences(text)
    if not sentences:
        return []
    sent_wc = [count_words(s) for s in sentences]
    chunks: List[str] = []
    start = 0

    while start < len(sentences):
        end = start
        total = 0
        while end < len(sentences):
            w = sent_wc[end]
            if total + w > max_words and end > start:
                break
            total += w
            end += 1

        if end == start:
            end = start + 1

        chunk = normalize_ws(" ".join(sentences[start:end]))
        if chunk:
            chunks.append(chunk)

        if end >= len(sentences):
            break

        # sentence-level overlap by words
        back = end - 1
        overlap = 0
        new_start = end
        while back >= start:
            overlap += sent_wc[back]
            new_start = back
            if overlap >= overlap_words:
                break
            back -= 1
        if new_start <= start:
            new_start = start + 1
        start = new_start

    return chunks


def adjacent_similarities(units: List[str]) -> List[float]:
    if len(units) < 2:
        return []
    if HAS_SKLEARN:
        vec = TfidfVectorizer(ngram_range=(1, 2), lowercase=True)
        mat = vec.fit_transform(units)
        return [float(cosine_similarity(mat[i], mat[i + 1])[0, 0]) for i in range(len(units) - 1)]

    # Fallback: simple token-frequency cosine without external deps.
    def token_counter(text: str) -> Counter:
        return Counter(WORD_RE.findall(text.lower()))

    def cosine_from_counter(a: Counter, b: Counter) -> float:
        if not a or not b:
            return 0.0
        common = set(a.keys()) & set(b.keys())
        dot = sum(a[t] * b[t] for t in common)
        norm_a = sqrt(sum(v * v for v in a.values()))
        norm_b = sqrt(sum(v * v for v in b.values()))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return dot / (norm_a * norm_b)

    vecs = [token_counter(u) for u in units]
    return [cosine_from_counter(vecs[i], vecs[i + 1]) for i in range(len(units) - 1)]


def percentile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    arr = sorted(values)
    idx = int(round((len(arr) - 1) * q))
    return arr[idx]


def semantic_refine(text: str) -> List[str]:
    units = split_paragraphs(text)
    if len(units) <= 1:
        units = split_sentences(text)
    if not units:
        return []
    if len(units) == 1:
        one = normalize_ws(units[0])
        return [one] if one else []

    sims = adjacent_similarities(units)
    low_threshold = percentile(sims, 0.25) if sims else 0.0

    groups: List[List[str]] = [[units[0]]]
    cur_words = count_words(units[0])
    unit_words = [count_words(x) for x in units]

    for i in range(1, len(units)):
        sim_prev = sims[i - 1] if i - 1 < len(sims) else 1.0
        this_words = unit_words[i]

        if cur_words >= SEM_MAX_WORDS:
            groups.append([units[i]])
            cur_words = this_words
            continue

        split_by_semantic = sim_prev <= low_threshold and cur_words >= SEM_MIN_WORDS
        split_by_target = cur_words >= SEM_TARGET_WORDS and sim_prev < 0.95
        if split_by_semantic or split_by_target:
            groups.append([units[i]])
            cur_words = this_words
        else:
            groups[-1].append(units[i])
            cur_words += this_words

    texts = [normalize_ws(" ".join(g)) for g in groups if g]
    texts = [t for t in texts if t]

    # Merge tiny fragments
    merged: List[str] = []
    for t in texts:
        if not merged:
            merged.append(t)
            continue
        if count_words(t) < SEM_ABSORB_BELOW:
            merged[-1] = normalize_ws(merged[-1] + " " + t)
        else:
            merged.append(t)
    return merged


def maybe_semantic_split(record: Dict[str, Any], text: str) -> List[str]:
    page_type = str(record.get("page_type", "")).strip().lower()
    section = str(record.get("section", "")).strip().lower()
    wc = count_words(text)

    if page_type == "recipe" and section in RECIPE_KEEP_SECTIONS and wc <= RECIPE_FORCE_SPLIT_WORDS:
        return [normalize_ws(text)] if text.strip() else []

    if wc <= SEM_TARGET_WORDS:
        return [normalize_ws(text)] if text.strip() else []

    refined = semantic_refine(text)
    return refined if refined else ([normalize_ws(text)] if text.strip() else [])


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_chunks(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    global_idx = 1

    for rec_idx, record in enumerate(records, start=1):
        text = str(record.get("text", "")).strip()
        if not text:
            continue

        parent_doc_id = make_parent_doc_id(record, rec_idx)
        title = str(record.get("title", ""))
        section = str(record.get("section", ""))
        source = str(record.get("source", ""))
        page_type = str(record.get("page_type", ""))
        cuisine = str(record.get("cuisine", "")) if record.get("cuisine") is not None else ""
        url = str(record.get("url", "")) if record.get("url") is not None else ""

        coarse_candidates = [text]
        coarse_method = "none"
        if count_words(text) > COARSE_TRIGGER_WORDS:
            coarse_candidates = fixed_size_split(
                text=text,
                max_words=COARSE_MAX_WORDS,
                overlap_words=COARSE_OVERLAP_WORDS,
            )
            coarse_method = "fixed_size_sentence"

        local_idx = 1
        for coarse_i, coarse_text in enumerate(coarse_candidates, start=1):
            refined_parts = maybe_semantic_split(record, coarse_text)
            for part_i, part in enumerate(refined_parts, start=1):
                wc = count_words(part)
                if wc == 0:
                    continue
                out.append(
                    {
                        "chunk_id": f"hyb_{global_idx:06d}",
                        "chunk_index": local_idx,
                        "parent_doc_id": parent_doc_id,
                        "source": source,
                        "page_type": page_type,
                        "title": title,
                        "section": section,
                        "cuisine": cuisine,
                        "url": url,
                        "text": part,
                        "word_count": wc,
                        "coarse_method": coarse_method,
                        "coarse_index": coarse_i,
                        "semantic_method": "tfidf_semantic" if len(refined_parts) > 1 else "none",
                        "semantic_index": part_i,
                    }
                )
                global_idx += 1
                local_idx += 1

    return out


def renumber_chunk_index(rows: List[Dict[str, Any]]) -> None:
    """Renumber chunk_index inside each parent_doc_id after merges."""
    counters: Dict[str, int] = {}
    for row in rows:
        parent = str(row.get("parent_doc_id", ""))
        counters[parent] = counters.get(parent, 0) + 1
        row["chunk_index"] = counters[parent]


def merge_final_fragments(rows: List[Dict[str, Any]], merge_below_words: int) -> List[Dict[str, Any]]:
    """
    Merge tiny chunks with priority:
    1) same section (within same title)
    2) same title
    Never merge chunks across different titles.
    """
    if not rows:
        return rows

    by_title: Dict[str, List[Dict[str, Any]]] = {}
    title_order: List[str] = []
    for row in rows:
        title_key = str(row.get("title", ""))
        if title_key not in by_title:
            by_title[title_key] = []
            title_order.append(title_key)
        by_title[title_key].append(dict(row))

    merged_all: List[Dict[str, Any]] = []

    for title_key in title_order:
        chunks = by_title[title_key]
        i = 0
        while i < len(chunks):
            cur = chunks[i]
            cur_wc = int(cur.get("word_count", 0))
            if cur_wc >= merge_below_words or len(chunks) == 1:
                i += 1
                continue

            cur_section = str(cur.get("section", ""))
            prev_idx = i - 1 if i - 1 >= 0 else None
            next_idx = i + 1 if i + 1 < len(chunks) else None

            # Priority 1: same section
            target_idx = None
            if prev_idx is not None and str(chunks[prev_idx].get("section", "")) == cur_section:
                target_idx = prev_idx
            elif next_idx is not None and str(chunks[next_idx].get("section", "")) == cur_section:
                target_idx = next_idx
            # Priority 2: same title (already guaranteed inside current group)
            elif prev_idx is not None:
                target_idx = prev_idx
            elif next_idx is not None:
                target_idx = next_idx

            if target_idx is None:
                i += 1
                continue

            target = chunks[target_idx]
            target_is_before = target_idx < i
            if target_is_before:
                target["text"] = normalize_ws(f"{target['text']} {cur['text']}")
            else:
                target["text"] = normalize_ws(f"{cur['text']} {target['text']}")
            target["word_count"] = count_words(target["text"])
            target["fragment_merged"] = True

            # Remove current tiny chunk.
            chunks.pop(i)
            # If merged into a previous chunk, continue from the previous index.
            if target_is_before:
                i = max(i - 1, 0)

        merged_all.extend(chunks)

    renumber_chunk_index(merged_all)
    return merged_all


def split_words_hard(text: str, max_words: int) -> List[str]:
    words = text.split()
    if not words:
        return []
    out: List[str] = []
    for i in range(0, len(words), max_words):
        part = " ".join(words[i : i + max_words]).strip()
        if part:
            out.append(part)
    return out


def split_text_to_range(text: str, min_words: int, target_words: int, max_words: int) -> List[str]:
    """
    Split one chunk into smaller chunks with sentence boundaries and hard cap.
    Tries to keep chunks around target_words and within [min_words, max_words].
    """
    sentences = split_sentences(text)
    if not sentences:
        return []

    sent_wc = [count_words(s) for s in sentences]
    out: List[str] = []
    cur: List[str] = []
    cur_wc = 0

    for sent, wc in zip(sentences, sent_wc):
        if wc > max_words:
            # Flush current and split this overlong sentence by words.
            if cur:
                out.append(normalize_ws(" ".join(cur)))
                cur = []
                cur_wc = 0
            out.extend(split_words_hard(sent, max_words))
            continue

        if cur_wc + wc > max_words and cur_wc >= min_words:
            out.append(normalize_ws(" ".join(cur)))
            cur = [sent]
            cur_wc = wc
            continue

        cur.append(sent)
        cur_wc += wc

        # Prefer cutting near target when safe.
        if cur_wc >= target_words and cur_wc >= min_words:
            out.append(normalize_ws(" ".join(cur)))
            cur = []
            cur_wc = 0

    if cur:
        if out and cur_wc < min_words:
            out[-1] = normalize_ws(out[-1] + " " + " ".join(cur))
        else:
            out.append(normalize_ws(" ".join(cur)))

    # Enforce hard max if previous merge created an oversized tail.
    final_out: List[str] = []
    for part in out:
        wc = count_words(part)
        if wc > max_words:
            final_out.extend(split_words_hard(part, max_words))
        else:
            final_out.append(part)

    return [p for p in final_out if p.strip()]


def renumber_chunk_ids(rows: List[Dict[str, Any]], prefix: str = "hyb") -> None:
    for i, row in enumerate(rows, start=1):
        row["chunk_id"] = f"{prefix}_{i:06d}"


def rebalance_chunk_lengths(
    rows: List[Dict[str, Any]],
    min_words: int = FINAL_MIN_WORDS,
    target_words: int = FINAL_TARGET_WORDS,
    max_words: int = FINAL_MAX_WORDS,
) -> List[Dict[str, Any]]:
    """
    Rebalance final chunk sizes inside each title.
    1) split chunks > max_words
    2) merge chunks < min_words with priority: same section -> same title
    Never merges across different titles.
    """
    if not rows:
        return rows

    by_title: Dict[str, List[Dict[str, Any]]] = {}
    title_order: List[str] = []
    for row in rows:
        title_key = str(row.get("title", ""))
        if title_key not in by_title:
            by_title[title_key] = []
            title_order.append(title_key)
        by_title[title_key].append(dict(row))

    def merge_short_chunks_with_priority(chunks: List[Dict[str, Any]], threshold: int) -> List[Dict[str, Any]]:
        i = 0
        while i < len(chunks):
            cur = chunks[i]
            cur_wc = int(cur.get("word_count", 0))
            if cur_wc >= threshold or len(chunks) == 1:
                i += 1
                continue

            cur_section = str(cur.get("section", ""))
            prev_idx = i - 1 if i - 1 >= 0 else None
            next_idx = i + 1 if i + 1 < len(chunks) else None

            target_idx = None
            if prev_idx is not None and str(chunks[prev_idx].get("section", "")) == cur_section:
                target_idx = prev_idx
            elif next_idx is not None and str(chunks[next_idx].get("section", "")) == cur_section:
                target_idx = next_idx
            elif prev_idx is not None:
                target_idx = prev_idx
            elif next_idx is not None:
                target_idx = next_idx

            if target_idx is None:
                i += 1
                continue

            target = chunks[target_idx]
            target_before = target_idx < i
            if target_before:
                merged_text = normalize_ws(f"{target['text']} {cur['text']}")
                target["text"] = merged_text
                target["word_count"] = count_words(merged_text)
                target["length_rebalanced"] = True
                chunks.pop(i)
                i = max(i - 1, 0)
            else:
                merged_text = normalize_ws(f"{cur['text']} {target['text']}")
                target["text"] = merged_text
                target["word_count"] = count_words(merged_text)
                target["length_rebalanced"] = True
                chunks.pop(i)
        return chunks

    rebalanced: List[Dict[str, Any]] = []

    for title_key in title_order:
        title_rows = by_title[title_key]

        # Pass 1: split oversized chunks.
        split_rows: List[Dict[str, Any]] = []
        for row in title_rows:
            text = str(row.get("text", "")).strip()
            wc = count_words(text)
            if wc <= max_words:
                row["word_count"] = wc
                split_rows.append(row)
                continue

            parts = split_text_to_range(text, min_words=min_words, target_words=target_words, max_words=max_words)
            for sub_idx, part in enumerate(parts, start=1):
                new_row = dict(row)
                new_row["text"] = part
                new_row["word_count"] = count_words(part)
                new_row["length_rebalanced"] = True
                new_row["length_split_index"] = sub_idx
                split_rows.append(new_row)

        # Pass 2: merge tiny chunks with section-first priority.
        split_rows = merge_short_chunks_with_priority(split_rows, min_words)

        # Pass 3: if merge created overlong chunk, split again.
        final_title_rows: List[Dict[str, Any]] = []
        for row in split_rows:
            text = str(row.get("text", "")).strip()
            wc = count_words(text)
            if wc <= max_words:
                row["word_count"] = wc
                final_title_rows.append(row)
                continue

            parts = split_text_to_range(text, min_words=min_words, target_words=target_words, max_words=max_words)
            for sub_idx, part in enumerate(parts, start=1):
                new_row = dict(row)
                new_row["text"] = part
                new_row["word_count"] = count_words(part)
                new_row["length_rebalanced"] = True
                new_row["length_split_index"] = sub_idx
                final_title_rows.append(new_row)

        # Pass 4: one more tiny merge after re-splitting.
        final_title_rows = merge_short_chunks_with_priority(final_title_rows, min_words)
        rebalanced.extend(final_title_rows)

    renumber_chunk_index(rebalanced)
    renumber_chunk_ids(rebalanced, prefix="hyb")
    return rebalanced


def strict_window_rebalance(
    rows: List[Dict[str, Any]],
    min_words: int = FINAL_MIN_WORDS,
    target_words: int = FINAL_TARGET_WORDS,
    max_words: int = FINAL_MAX_WORDS,
    max_iterations: int = STRICT_MAX_ITERATIONS,
) -> List[Dict[str, Any]]:
    """
    Aggressive post-processing to pull chunks into [min_words, max_words].
    Rules:
    - Never merge across different titles.
    - Prefer same-section merge for short chunks.
    - Split oversized chunks using sentence-aware splitter.
    """
    if not rows:
        return rows

    def split_oversized(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for row in chunks:
            text = str(row.get("text", "")).strip()
            wc = count_words(text)
            if wc <= max_words:
                row["word_count"] = wc
                out.append(row)
                continue
            parts = split_text_to_range(text, min_words=min_words, target_words=target_words, max_words=max_words)
            for sub_idx, part in enumerate(parts, start=1):
                new_row = dict(row)
                new_row["text"] = part
                new_row["word_count"] = count_words(part)
                new_row["strict_rebalanced"] = True
                new_row["strict_split_index"] = sub_idx
                out.append(new_row)
        return out

    def choose_merge_target_idx(chunks: List[Dict[str, Any]], i: int) -> int | None:
        cur = chunks[i]
        cur_section = str(cur.get("section", ""))
        prev_idx = i - 1 if i - 1 >= 0 else None
        next_idx = i + 1 if i + 1 < len(chunks) else None

        candidates: List[int] = []
        if prev_idx is not None and str(chunks[prev_idx].get("section", "")) == cur_section:
            candidates.append(prev_idx)
        if next_idx is not None and str(chunks[next_idx].get("section", "")) == cur_section:
            candidates.append(next_idx)
        if not candidates:
            if prev_idx is not None:
                candidates.append(prev_idx)
            if next_idx is not None:
                candidates.append(next_idx)
        if not candidates:
            return None

        cur_wc = int(cur.get("word_count", 0))
        best_idx = candidates[0]
        best_score = float("inf")
        for idx in candidates:
            merged_wc = int(chunks[idx].get("word_count", 0)) + cur_wc
            overflow_penalty = max(0, merged_wc - max_words) * 3
            score = abs(merged_wc - target_words) + overflow_penalty
            if score < best_score:
                best_score = score
                best_idx = idx
        return best_idx

    by_title: Dict[str, List[Dict[str, Any]]] = {}
    title_order: List[str] = []
    for row in rows:
        title_key = str(row.get("title", ""))
        if title_key not in by_title:
            by_title[title_key] = []
            title_order.append(title_key)
        by_title[title_key].append(dict(row))

    final_rows: List[Dict[str, Any]] = []

    for title_key in title_order:
        chunks = by_title[title_key]
        chunks = split_oversized(chunks)

        for _ in range(max_iterations):
            changed = False

            # Merge undersized chunks
            i = 0
            while i < len(chunks):
                cur_wc = int(chunks[i].get("word_count", 0))
                if cur_wc >= min_words or len(chunks) == 1:
                    i += 1
                    continue

                target_idx = choose_merge_target_idx(chunks, i)
                if target_idx is None:
                    i += 1
                    continue

                cur = chunks[i]
                target = chunks[target_idx]
                target_before = target_idx < i
                if target_before:
                    merged_text = normalize_ws(f"{target['text']} {cur['text']}")
                    target["text"] = merged_text
                    target["word_count"] = count_words(merged_text)
                    target["strict_rebalanced"] = True
                    chunks.pop(i)
                    i = max(i - 1, 0)
                else:
                    merged_text = normalize_ws(f"{cur['text']} {target['text']}")
                    target["text"] = merged_text
                    target["word_count"] = count_words(merged_text)
                    target["strict_rebalanced"] = True
                    chunks.pop(i)
                changed = True

            # Split oversize again if merge produced long chunk.
            new_chunks = split_oversized(chunks)
            if len(new_chunks) != len(chunks):
                changed = True
            chunks = new_chunks

            if not changed:
                break

        final_rows.extend(chunks)

    renumber_chunk_index(final_rows)
    renumber_chunk_ids(final_rows, prefix="hyb")
    return final_rows


def pack_title_chunks_to_window(
    rows: List[Dict[str, Any]],
    min_words: int = FINAL_MIN_WORDS,
    target_words: int = FINAL_TARGET_WORDS,
    max_words: int = FINAL_MAX_WORDS,
) -> List[Dict[str, Any]]:
    """
    Final compaction pass within each title to tighten length distribution.
    Never merges across titles.
    """
    if not rows:
        return rows

    by_title: Dict[str, List[Dict[str, Any]]] = {}
    title_order: List[str] = []
    for row in rows:
        title = str(row.get("title", ""))
        if title not in by_title:
            by_title[title] = []
            title_order.append(title)
        by_title[title].append(dict(row))

    packed_all: List[Dict[str, Any]] = []

    def merge_rows(group: List[Dict[str, Any]]) -> Dict[str, Any]:
        base = dict(group[0])
        text = normalize_ws(" ".join(str(x.get("text", "")) for x in group))
        base["text"] = text
        base["word_count"] = count_words(text)
        base["final_packed"] = True
        sections = [str(x.get("section", "")) for x in group]
        uniq_sections = list(dict.fromkeys(sections))
        if len(uniq_sections) > 1:
            base["section"] = uniq_sections[0]
            base["mixed_sections"] = uniq_sections
        return base

    for title in title_order:
        items = by_title[title]
        local_out: List[Dict[str, Any]] = []
        buffer: List[Dict[str, Any]] = []
        buffer_wc = 0

        def flush_buffer() -> None:
            nonlocal buffer, buffer_wc
            if not buffer:
                return
            local_out.append(merge_rows(buffer))
            buffer = []
            buffer_wc = 0

        for row in items:
            text = str(row.get("text", "")).strip()
            wc = count_words(text)
            row["word_count"] = wc

            # Split oversized unit first.
            units: List[Dict[str, Any]] = []
            if wc > max_words:
                parts = split_text_to_range(text, min_words=min_words, target_words=target_words, max_words=max_words)
                for part in parts:
                    new_row = dict(row)
                    new_row["text"] = part
                    new_row["word_count"] = count_words(part)
                    units.append(new_row)
            else:
                units = [row]

            for unit in units:
                uwc = int(unit.get("word_count", 0))
                tentative = buffer_wc + uwc

                if not buffer:
                    buffer = [unit]
                    buffer_wc = uwc
                    if min_words <= buffer_wc <= max_words:
                        flush_buffer()
                    continue

                if tentative < min_words:
                    buffer.append(unit)
                    buffer_wc = tentative
                    continue

                if min_words <= tentative <= max_words:
                    buffer.append(unit)
                    buffer_wc = tentative
                    flush_buffer()
                    continue

                # tentative > max_words
                if buffer_wc >= min_words:
                    # Buffer is already valid; close it and retry this unit.
                    flush_buffer()
                    buffer = [unit]
                    buffer_wc = uwc
                    if min_words <= buffer_wc <= max_words:
                        flush_buffer()
                else:
                    # Buffer too short: accept overflow once to avoid tiny chunk.
                    buffer.append(unit)
                    buffer_wc = tentative
                    flush_buffer()

        # Handle tail
        if buffer:
            if buffer_wc < min_words and local_out:
                prev = local_out.pop()
                merged = merge_rows([prev] + buffer)
                # If this tail merge is too long, split once by target window.
                if int(merged.get("word_count", 0)) > max_words:
                    parts = split_text_to_range(
                        str(merged["text"]),
                        min_words=min_words,
                        target_words=target_words,
                        max_words=max_words,
                    )
                    for part in parts:
                        row_new = dict(merged)
                        row_new["text"] = part
                        row_new["word_count"] = count_words(part)
                        local_out.append(row_new)
                else:
                    local_out.append(merged)
            else:
                flush_buffer()

        packed_all.extend(local_out)

    renumber_chunk_index(packed_all)
    renumber_chunk_ids(packed_all, prefix="hyb")
    return packed_all


def print_stats(chunks: List[Dict[str, Any]]) -> None:
    print(f"Input: {INPUT_FILE}")
    print(f"Output: {OUTPUT_FILE}")
    print(f"Chunks: {len(chunks)}")
    if not chunks:
        return
    wc = [int(c["word_count"]) for c in chunks]
    print(f"Word count min/max: {min(wc)}/{max(wc)}")
    print(f"Word count p50: {sorted(wc)[len(wc)//2]}")
    coarse_count = sum(1 for c in chunks if c.get("coarse_method") != "none")
    semantic_count = sum(1 for c in chunks if c.get("semantic_method") != "none")
    print(f"Chunks from coarse split: {coarse_count}")
    print(f"Chunks from semantic split: {semantic_count}")
    in_target = sum(1 for x in wc if FINAL_MIN_WORDS <= x <= FINAL_MAX_WORDS)
    print(
        f"In [{FINAL_MIN_WORDS}, {FINAL_MAX_WORDS}] words: "
        f"{in_target}/{len(wc)} ({(in_target / len(wc)):.1%})"
    )


def save_word_count_histogram(chunks: List[Dict[str, Any]], output_path: Path, bins: int = HIST_BINS) -> None:
    if not chunks:
        return

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required to generate histogram. "
            "Please install it in the current environment."
        ) from exc

    wc = [int(c.get("word_count", 0)) for c in chunks]
    wc = [x for x in wc if x >= 0]
    if not wc:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.hist(wc, bins=bins, color="#4c72b0", edgecolor="#ffffff", alpha=0.9)
    plt.title("Chunk Word Count Distribution")
    plt.xlabel("Word Count")
    plt.ylabel("Frequency")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")
    records = parse_records(INPUT_FILE)
    chunks = build_chunks(records)
    chunks = merge_final_fragments(chunks, FINAL_FRAGMENT_MERGE_BELOW)
    chunks = rebalance_chunk_lengths(
        chunks,
        min_words=FINAL_MIN_WORDS,
        target_words=FINAL_TARGET_WORDS,
        max_words=FINAL_MAX_WORDS,
    )
    if STRICT_WINDOW_REBALANCE:
        chunks = strict_window_rebalance(
            chunks,
            min_words=FINAL_MIN_WORDS,
            target_words=FINAL_TARGET_WORDS,
            max_words=FINAL_MAX_WORDS,
            max_iterations=STRICT_MAX_ITERATIONS,
        )
    if FINAL_TITLE_PACKING:
        chunks = pack_title_chunks_to_window(
            chunks,
            min_words=FINAL_MIN_WORDS,
            target_words=FINAL_TARGET_WORDS,
            max_words=FINAL_MAX_WORDS,
        )
    if POST_PACK_STRICT_REBALANCE:
        chunks = strict_window_rebalance(
            chunks,
            min_words=FINAL_MIN_WORDS,
            target_words=FINAL_TARGET_WORDS,
            max_words=FINAL_MAX_WORDS,
            max_iterations=1,
        )
    write_jsonl(OUTPUT_FILE, chunks)
    save_word_count_histogram(chunks, HISTOGRAM_FILE, bins=HIST_BINS)
    print_stats(chunks)
    print(f"Histogram: {HISTOGRAM_FILE}")


if __name__ == "__main__":
    main()
