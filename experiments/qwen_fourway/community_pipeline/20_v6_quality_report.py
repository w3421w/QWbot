#!/usr/bin/env python3
"""QWbot V6 数据质量统计（只读本地 JSONL，不联网、不下载）。"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def _error(errors: list[dict[str, Any]], path: Path, line_number: int, message: str) -> None:
    errors.append({"file": str(path), "line_number": line_number, "error": message})


def _read_jsonl(path: str | Path, errors: list[dict[str, Any]]) -> list[tuple[int, Any]]:
    path = Path(path)
    rows: list[tuple[int, Any]] = []
    try:
        handle = path.open("r", encoding="utf-8-sig")
    except Exception as exc:
        _error(errors, path, 0, str(exc) or exc.__class__.__name__)
        return rows
    with handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                _error(errors, path, line_number, "空行")
                continue
            try:
                rows.append((line_number, json.loads(line)))
            except Exception as exc:
                _error(errors, path, line_number, str(exc) or "JSON解析失败")
    return rows


def _dataset_summary(path: str | Path, errors: list[dict[str, Any]]) -> dict[str, Any]:
    rows = _read_jsonl(path, errors)
    total = len(rows)
    valid = empty = 0
    speakers = turns = text_length = 0
    for line_number, row in rows:
        if not isinstance(row, dict):
            _error(errors, Path(path), line_number, "记录必须是对象")
            continue
        dialogue = row.get("dialogue")
        if not dialogue:
            empty += 1
            continue
        if not isinstance(dialogue, list) or any(
                not isinstance(item, dict) or not isinstance(item.get("speaker"), str)
                or not isinstance(item.get("text"), str) for item in dialogue):
            _error(errors, Path(path), line_number, "dialogue格式异常")
            continue
        valid += 1
        turns += len(dialogue)
        speakers += len({item["speaker"] for item in dialogue})
        text_length += sum(len(item["text"]) for item in dialogue)
    return {"total_records": total, "valid_records": valid, "empty_records": empty,
            "average_speaker_count": speakers / valid if valid else 0,
            "average_rounds": turns / valid if valid else 0,
            "average_text_length": text_length / valid if valid else 0}


def _classification_summary(path: str | Path, errors: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    role, style, community = Counter(), Counter(), Counter()
    for line_number, row in _read_jsonl(path, errors):
        if not isinstance(row, dict):
            _error(errors, Path(path), line_number, "分类记录必须是对象")
            continue
        data_role = row.get("data_role")
        if isinstance(data_role, str) and data_role:
            role[data_role] += 1
        else:
            _error(errors, Path(path), line_number, "data_role缺失或类型错误")
        tags = row.get("style_tags", [])
        if isinstance(tags, list):
            for tag in tags:
                if isinstance(tag, str) and tag:
                    style[tag] += 1
                else:
                    _error(errors, Path(path), line_number, "style_tags包含异常值")
        else:
            _error(errors, Path(path), line_number, "style_tags必须是数组")
        hint = row.get("community_hint")
        if isinstance(hint, str) and hint:
            community[hint] += 1
    role_distribution = {name: role.get(name, 0)
                         for name in ("finetune", "knowledge", "discard", "review")}
    return {"role_distribution": role_distribution, "style_distribution": dict(style),
            "community_distribution": dict(community)}


def generate_quality_report(raw_path: str | Path, classified_path: str | Path,
                            output_path: str | Path = "v6_quality_report.json") -> dict[str, Any]:
    errors: list[dict[str, Any]] = []
    dataset = _dataset_summary(raw_path, errors)
    distributions = _classification_summary(classified_path, errors)
    report = {"dataset_summary": dataset, **distributions, "errors": errors}
    Path(output_path).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw", help="v6_dialogue_raw.jsonl")
    parser.add_argument("classified", help="19号分类结果 JSONL")
    parser.add_argument("--output", default="v6_quality_report.json")
    args = parser.parse_args()
    print(json.dumps(generate_quality_report(args.raw, args.classified, args.output), ensure_ascii=False))


if __name__ == "__main__":
    main()
