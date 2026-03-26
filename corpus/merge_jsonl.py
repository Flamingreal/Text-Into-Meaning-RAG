#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


DEFAULT_FILES = ["80Cuisine.jsonl", "wikibook.jsonl", "wikipedia.jsonl"]
DEFAULT_OUTPUT = "corpus.jsonl"
WORD_RE = re.compile(r"\b[\w'-]+\b")


def iter_non_empty_lines(path: Path) -> Iterable[Tuple[str, Dict]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                obj = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path.name} 第 {line_no} 行不是合法 JSON: {exc}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"{path.name} 第 {line_no} 行不是 JSON 对象。")
            yield text, obj


def count_words(text: str) -> int:
    return len(WORD_RE.findall(text))


def save_wordcount_histograms(
    counts_by_file: Dict[str, List[int]],
    output_path: Path,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("跳过直方图：当前环境未安装 matplotlib。")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    file_names = list(counts_by_file.keys())
    if not file_names:
        return

    # Figure 1: combined histogram for quick comparison.
    plt.figure(figsize=(10, 6))
    for name in file_names:
        data = counts_by_file[name]
        if not data:
            continue
        plt.hist(data, bins=40, alpha=0.35, label=name, edgecolor="white")
    plt.title("Word Count Distribution by JSONL")
    plt.xlabel("Word Count (text field)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"已生成直方图: {output_path}")


def merge_jsonl(input_dir: Path, input_files: List[str], output_file: str) -> None:
    output_path = input_dir / output_file
    hist_path = input_dir / f"{Path(output_file).stem}_wordcount_hist.png"
    total = 0
    counts_by_file: Dict[str, List[int]] = {}

    with output_path.open("w", encoding="utf-8", newline="\n") as out:
        for file_name in input_files:
            src = input_dir / file_name
            if not src.exists():
                raise FileNotFoundError(f"找不到输入文件: {src}")

            file_count = 0
            word_counts: List[int] = []
            for line_text, obj in iter_non_empty_lines(src):
                out.write(line_text)
                out.write("\n")
                file_count += 1
                text_field = str(obj.get("text", "")) if obj.get("text") is not None else ""
                word_counts.append(count_words(text_field))

            total += file_count
            counts_by_file[file_name] = word_counts
            print(f"{file_name}: {file_count} 条")

    print(f"合并完成: {output_path}")
    print(f"总条数: {total}")
    save_wordcount_histograms(counts_by_file, hist_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="合并 corpus 目录下多个 JSONL 文件。")
    parser.add_argument(
        "--input-dir",
        default="corpus",
        help="输入目录（默认: corpus）",
    )
    parser.add_argument(
        "--files",
        nargs="+",
        default=DEFAULT_FILES,
        help=f"输入文件列表（默认: {' '.join(DEFAULT_FILES)}）",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"输出文件名（默认: {DEFAULT_OUTPUT}）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    merge_jsonl(Path(args.input_dir), list(args.files), args.output)


if __name__ == "__main__":
    main()
