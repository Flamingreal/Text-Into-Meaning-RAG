#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

# ----------------------------
# Config (edit here directly)
# ----------------------------
ROOT_DIR = Path(__file__).resolve().parents[1]
INPUT_FILE = (
    ROOT_DIR
    / "artifacts"
    / "chunking"
    / "fixed_size"
    / "corpus_fixed.jsonl"
)
OUTPUT_FILE = (
    ROOT_DIR
    / "artifacts"
    / "chunking"
    / "fixed_size"
    / "corpus_embedding.jsonl"
)

# Required embedding model
MODEL_NAME = "Qwen/Qwen3-Embedding-4B"
# Optional local model directory. Set this if your environment
# cannot access Hugging Face directly.
# Example:
# LOCAL_MODEL_DIR = ROOT_DIR / "models" / "Qwen3-Embedding-4B"
LOCAL_MODEL_DIR: Path | None = None
MODEL_CHECKPOINTS_DIR = ROOT_DIR / "model_checkpoints"
PERSIST_MODEL_TO_CHECKPOINT = True

# Inference controls
BATCH_SIZE = 4
MAX_LENGTH = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

# Semantic subchunk length constraints (in words)
MIN_SUBCHUNK_WORDS = 40
TARGET_SUBCHUNK_WORDS = 80
MAX_SUBCHUNK_WORDS = 120

# Adjacent sentence semantic break thresholds (embedding cosine)
HARD_BREAK_SIM = 0.45
SOFT_BREAK_SIM = 0.60

SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n{2,}")
WORD_RE = re.compile(r"\b[\w'-]+\b")


def _has_model_weights(model_dir: Path) -> bool:
    direct_files = [
        "model.safetensors",
        "pytorch_model.bin",
        "pytorch_model.bin.index.json",
        "model.safetensors.index.json",
    ]
    if any((model_dir / name).exists() for name in direct_files):
        return True

    if list(model_dir.glob("*.safetensors")):
        return True
    if list(model_dir.glob("pytorch_model-*.bin")):
        return True
    return False


def normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def split_sentences(text: str) -> List[str]:
    text = text.strip()
    if not text:
        return []
    return [s for s in (normalize_ws(p) for p in SENTENCE_SPLIT_RE.split(text)) if s]


def word_count(text: str) -> int:
    return len(WORD_RE.findall(text))


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


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    denom = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / denom


def encode_texts(
    texts: List[str], tokenizer: AutoTokenizer, model: AutoModel, batch_size: int = BATCH_SIZE
) -> torch.Tensor:
    if not texts:
        return torch.empty((0, 1), dtype=torch.float32)

    all_embs: List[torch.Tensor] = []
    model.eval()

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            )
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            outputs = model(**inputs)
            pooled = mean_pool(outputs.last_hidden_state, inputs["attention_mask"])
            pooled = F.normalize(pooled, p=2, dim=1)
            all_embs.append(pooled.detach().cpu())

    return torch.cat(all_embs, dim=0)


def adjacent_similarities(sentences: List[str], tokenizer: AutoTokenizer, model: AutoModel) -> List[float]:
    if len(sentences) < 2:
        return []
    emb = encode_texts(sentences, tokenizer, model, batch_size=BATCH_SIZE)
    sims = F.cosine_similarity(emb[:-1], emb[1:], dim=1)
    return sims.tolist()


def semantic_split_sentences(
    sentences: List[str], tokenizer: AutoTokenizer, model: AutoModel
) -> List[str]:
    if not sentences:
        return []
    if len(sentences) == 1:
        return [sentences[0]]

    sims = adjacent_similarities(sentences, tokenizer, model)
    sent_wc = [word_count(s) for s in sentences]

    groups: List[List[str]] = [[sentences[0]]]
    current_words = sent_wc[0]

    for i in range(1, len(sentences)):
        sim_prev = sims[i - 1] if i - 1 < len(sims) else 1.0
        this_words = sent_wc[i]

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


def build_subchunks(rows: List[Dict], tokenizer: AutoTokenizer, model: AutoModel) -> List[Dict]:
    out: List[Dict] = []
    global_idx = 1

    for row in rows:
        parent_id = str(row.get("chunk_id", "unknown_parent"))
        text = str(row.get("text", "")).strip()
        sentences = split_sentences(text)
        if not sentences:
            continue

        sub_texts = semantic_split_sentences(sentences, tokenizer, model)
        for sub_idx, sub_text in enumerate(sub_texts, start=1):
            new_row = dict(row)
            new_row["parent_chunk_id"] = parent_id
            new_row["chunk_id"] = f"emb_{global_idx:06d}"
            new_row["subchunk_index"] = sub_idx
            new_row["split_method"] = "embedding_cosine"
            new_row["embedding_model"] = MODEL_NAME
            new_row["text"] = sub_text
            new_row["word_count"] = word_count(sub_text)
            out.append(new_row)
            global_idx += 1

    return out


def persist_model_checkpoint(model_name: str, checkpoints_dir: Path) -> Path:
    """
    Download and persist model/tokenizer into local model_checkpoints directory.
    If already present, reuse without downloading.
    """
    canonical_dir = checkpoints_dir / model_name.replace("/", "__")
    legacy_dir = checkpoints_dir / model_name.split("/")[-1]
    for candidate in [canonical_dir, legacy_dir]:
        if candidate.exists() and _has_model_weights(candidate):
            return candidate

    if canonical_dir.exists() and not _has_model_weights(canonical_dir):
        shutil.rmtree(canonical_dir)
    canonical_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading model to local checkpoint: {canonical_dir}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_name,
        trust_remote_code=True,
        dtype=TORCH_DTYPE,
    )
    tokenizer.save_pretrained(canonical_dir)
    model.save_pretrained(canonical_dir)
    return canonical_dir


def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    if LOCAL_MODEL_DIR is not None:
        model_ref = str(LOCAL_MODEL_DIR)
    elif PERSIST_MODEL_TO_CHECKPOINT:
        model_ref = str(persist_model_checkpoint(MODEL_NAME, MODEL_CHECKPOINTS_DIR))
    else:
        model_ref = MODEL_NAME

    print(f"Loading model: {model_ref}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_ref, trust_remote_code=True)
        model = AutoModel.from_pretrained(
            model_ref,
            trust_remote_code=True,
            dtype=TORCH_DTYPE,
        ).to(DEVICE)
    except Exception as exc:
        raise RuntimeError(
            "Failed to load embedding model. "
            "If network is restricted, set LOCAL_MODEL_DIR to a local Qwen3-Embedding-4B path."
        ) from exc

    parent_rows = load_jsonl(INPUT_FILE)
    subchunk_rows = build_subchunks(parent_rows, tokenizer, model)
    write_jsonl(OUTPUT_FILE, subchunk_rows)

    print(f"Device: {DEVICE}")
    print(f"Input: {INPUT_FILE}")
    print(f"Output: {OUTPUT_FILE}")
    print(f"Parent chunks: {len(parent_rows)}")
    print(f"Subchunks: {len(subchunk_rows)}")
    if subchunk_rows:
        wc = [int(r["word_count"]) for r in subchunk_rows]
        print(f"Word count min/max: {min(wc)}/{max(wc)}")


if __name__ == "__main__":
    main()
