#!/usr/bin/env python3
"""QWbot V6 SFT 去重与可复现多标签分层划分（仅使用标准库）。"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

SPLITS = ("train", "validation", "test")


def conversation_hash(row: dict[str, Any]) -> str:
    """仅以按原顺序排列的 role/content 计算 SHA256，不包含 metadata。"""
    conversation = [[message["role"], message["content"]] for message in row["messages"]]
    encoded = json.dumps(conversation, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _labels(row: dict[str, Any]) -> set[tuple[str, str]]:
    metadata = row.get("metadata") or {}
    labels = set()
    for field in ("source", "community_hint"):
        value = metadata.get(field)
        labels.add((field, value if isinstance(value, str) and value else "__missing__"))
    tags = metadata.get("style_tags", [])
    styles = {tag for tag in tags if isinstance(tag, str) and tag} if isinstance(tags, list) else set()
    labels.update(("style_tags", tag) for tag in (styles or {"__missing__"}))
    return labels


def _stratify(rows: list[dict[str, Any]], ratios: tuple[float, ...], seed: int) -> list[list[dict[str, Any]]]:
    """优先处理剩余样本最少的标签，按标签配额缺口分配，严格遵守集合容量。"""
    rng = random.Random(seed)
    exact = [len(rows) * ratio for ratio in ratios]
    capacity = [math.floor(value) for value in exact]
    for index in sorted(range(3), key=lambda i: (-(exact[i] - capacity[i]), i))[:len(rows) - sum(capacity)]:
        capacity[index] += 1
    remaining = capacity[:]
    labels = [_labels(row) for row in rows]
    members: dict[tuple[str, str], set[int]] = defaultdict(set)
    for index, row_labels in enumerate(labels):
        for label in sorted(row_labels):
            members[label].add(index)
    targets = {label: [len(indices) * ratio for ratio in ratios] for label, indices in members.items()}
    assigned = [Counter() for _ in SPLITS]
    output: list[list[dict[str, Any]]] = [[] for _ in SPLITS]
    # Seeded priorities avoid set iteration order and keep ties reproducible.
    order = list(range(len(rows)))
    rng.shuffle(order)
    priority = {index: rank for rank, index in enumerate(order)}
    while members:
        label = min(members, key=lambda item: (len(members[item]), item))
        for index in sorted(members[label], key=priority.__getitem__):
            choices = [i for i in range(3) if remaining[i] > 0]
            rng.shuffle(choices)
            def score(i: int) -> tuple[float, float, int]:
                return (targets[label][i] - assigned[i][label],
                        sum(targets[tag][i] - assigned[i][tag] for tag in sorted(labels[index])),
                        remaining[i])
            chosen = max(choices, key=score)
            output[chosen].append(rows[index])
            remaining[chosen] -= 1
            assigned[chosen].update(labels[index])
            for tag in sorted(labels[index]):
                members[tag].remove(index)
                if not members[tag]:
                    del members[tag]
    for items in output:
        rng.shuffle(items)
    return output


def split_dataset(input_path: str | Path = "qwen_sft_train.jsonl",
                  output_dir: str | Path = ".", train_ratio: float = 0.9,
                  valid_ratio: float = 0.05, test_ratio: float = 0.05,
                  seed: int = 42) -> dict[str, Any]:
    """total_input 包含全部输入行；total 仅包含合法且去重后的样本。

    分布格式为 {集合名: {标签: 样本数}}，缺失标签计入 __missing__。
    重复 style 标签每条样本只计一次；原始记录（含额外字段）完整保留。
    """
    ratios = (train_ratio, valid_ratio, test_ratio)
    if any(not math.isfinite(ratio) or ratio < 0 or ratio > 1 for ratio in ratios) or not math.isclose(sum(ratios), 1, abs_tol=1e-9):
        raise ValueError("划分比例必须在 [0, 1] 内且总和为 1")
    input_path, output_dir = Path(input_path), Path(output_dir)
    names = [f"{name}.jsonl" for name in SPLITS] + ["duplicate_report.json", "split_report.json", "errors.jsonl"]
    if input_path.resolve() in {(output_dir / name).resolve() for name in names}:
        raise ValueError("输入文件不能与输出文件相同")
    rows, errors, seen = [], [], set()
    total_input = duplicates = 0
    with input_path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            total_input += 1
            try:
                row = json.loads(line)
                if not isinstance(row, dict) or not isinstance(row.get("messages"), list) or not row["messages"]:
                    raise ValueError("记录必须包含非空 messages 数组")
                if any(not isinstance(message, dict) or not isinstance(message.get("role"), str)
                       or not message["role"].strip() or not isinstance(message.get("content"), str)
                       for message in row["messages"]):
                    raise ValueError("每条 message 必须包含有效 role 和字符串 content")
                if row.get("metadata") is not None and not isinstance(row["metadata"], dict):
                    raise ValueError("metadata 必须是对象或 null")
                digest = conversation_hash(row)
            except (ValueError, TypeError) as exc:
                errors.append({"file": str(input_path), "line_number": line_number, "error": str(exc)})
                continue
            if digest in seen:
                duplicates += 1
            else:
                seen.add(digest)
                rows.append(row)
    groups = _stratify(rows, ratios, seed)
    report = {"total": len(rows), **{f"{name}_count": len(group) for name, group in zip(SPLITS, groups)}}
    for field, key in (("source", "source_distribution"), ("community_hint", "community_distribution"), ("style_tags", "style_distribution")):
        report[key] = {name: dict(sorted(Counter(value for row in group for kind, value in _labels(row) if kind == field).items()))
                       for name, group in zip(SPLITS, groups)}
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, group in zip(SPLITS, groups):
        (output_dir / f"{name}.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in group), encoding="utf-8")
    (output_dir / "errors.jsonl").write_text("".join(json.dumps(error, ensure_ascii=False) + "\n" for error in errors), encoding="utf-8")
    duplicate_report = {"total_input": total_input, "unique_count": len(rows), "duplicate_count": duplicates}
    for name, value in (("duplicate_report.json", duplicate_report), ("split_report.json", report)):
        (output_dir / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", default="qwen_sft_train.jsonl")
    parser.add_argument("--output-dir", default=".")
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--valid-ratio", type=float, default=0.05)
    parser.add_argument("--test-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        report = split_dataset(args.input, args.output_dir, args.train_ratio, args.valid_ratio, args.test_ratio, args.seed)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
