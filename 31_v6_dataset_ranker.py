#!/usr/bin/env python3
"""Rank real datasets in dataset_inventory.json for QWbot V6.

This is a selection/reporting utility only; it does not create or alter a
training pipeline.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable

FIELD_GROUPS = {
    "dialogue/conversation/messages": ("dialogue", "conversation", "messages"),
    "comment/reply/content": ("comment", "reply", "content"),
    "user/author": ("user", "author"),
}
PATH_KEYWORDS = ("贴吧", "抗吧", "bilibili", "comment", "forum", "community")


def _strings(value: Any) -> Iterable[str]:
    if isinstance(value, dict):
        for key, child in value.items():
            yield str(key).lower()
            yield from _strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _strings(child)
    elif isinstance(value, str):
        yield value.lower()


def _has_field(entry: dict[str, Any], names: tuple[str, ...]) -> bool:
    """Match field names in observed/probe metadata without inspecting data files."""
    needles = tuple(n.lower() for n in names)
    for value in _strings(entry):
        # Match dotted paths and ordinary field names, while avoiding prose-only
        # suggested adapter descriptions as the sole source of a false match.
        parts = value.replace("[]", ".").replace("/", ".").split(".")
        if any(part in needles for part in parts):
            return True
    return False


def _large_cutoff(entries: list[dict[str, Any]]) -> float:
    sizes = sorted(float(e.get("size", 0) or 0) for e in entries)
    if not sizes:
        return math.inf
    # Relative threshold makes the ranker useful for both tiny fixtures and a
    # large inventory; at least 100 KiB avoids calling one-byte fixtures large.
    index = max(0, math.ceil(len(sizes) * 0.75) - 1)
    return max(100 * 1024, sizes[index])


def rank_datasets(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cutoff = _large_cutoff(entries)
    ranked = []
    for entry in entries:
        file_name = str(entry.get("file", ""))
        lowered = file_name.lower()
        score = 0
        reasons: list[str] = []
        if _has_field(entry, FIELD_GROUPS["dialogue/conversation/messages"]):
            score += 5; reasons.append("存在 dialogue/conversation/messages 字段 (+5)")
        if _has_field(entry, FIELD_GROUPS["comment/reply/content"]):
            score += 5; reasons.append("存在 comment/reply/content 字段 (+5)")
        if _has_field(entry, FIELD_GROUPS["user/author"]):
            score += 3; reasons.append("存在 user/author 字段 (+3)")
        size = float(entry.get("size", 0) or 0)
        if size >= cutoff:
            score += 3; reasons.append("文件较大 (+3)")
        if str(entry.get("format", "")).lower() == "jsonl" or lowered.endswith(".jsonl"):
            score += 3; reasons.append("JSONL 格式 (+3)")
        matched = [word for word in PATH_KEYWORDS if word in lowered]
        if matched:
            score += 5; reasons.append("路径/文件名包含: " + ", ".join(matched) + " (+5)")

        if _has_field(entry, FIELD_GROUPS["dialogue/conversation/messages"]):
            role = "finetune"
        elif _has_field(entry, FIELD_GROUPS["comment/reply/content"]):
            role = "review" if score < 8 else "finetune"
        elif size >= cutoff:
            role = "knowledge"
        else:
            role = "review"
        ranked.append({"file": file_name, "score": score, "reason": reasons, "suggested_role": role})
    return sorted(ranked, key=lambda x: (-x["score"], x["file"]))


def write_report(ranked: list[dict[str, Any]], output_json: Path, output_md: Path) -> None:
    output_json.write_text(json.dumps(ranked, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# QWbot V6 数据集候选", "", "按评分降序列出前 20 个候选：", "", "| 排名 | 文件 | 分数 | 建议用途 | 原因 |", "|---:|---|---:|---|---|"]
    for i, item in enumerate(ranked[:20], 1):
        reasons = "；".join(item["reason"]) or "无命中规则"
        lines.append(f"| {i} | `{item['file']}` | {item['score']} | {item['suggested_role']} | {reasons} |")
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, default=Path("dataset_inventory.json"))
    parser.add_argument("--output-json", type=Path, default=Path("candidate_datasets.json"))
    parser.add_argument("--output-md", type=Path, default=Path("top_candidates.md"))
    args = parser.parse_args()
    entries = json.loads(args.inventory.read_text(encoding="utf-8"))
    if not isinstance(entries, list):
        raise SystemExit("dataset inventory must be a JSON array")
    ranked = rank_datasets(entries)
    write_report(ranked, args.output_json, args.output_md)
    print(f"ranked {len(ranked)} datasets; wrote {args.output_json} and {args.output_md}")


if __name__ == "__main__":
    main()
