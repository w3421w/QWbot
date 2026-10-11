#!/usr/bin/env python3
"""将 QWbot V6 finetune 对话导出为 Qwen SFT JSONL。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read_jsonl(path: str | Path, errors: list[dict[str, Any]], kind: str) -> list[tuple[int, Any]]:
    path = Path(path)
    rows: list[tuple[int, Any]] = []
    try:
        handle = path.open("r", encoding="utf-8-sig")
    except Exception as exc:
        errors.append({"file": str(path), "line_number": 0, "error": f"{kind}文件无法读取: {exc}"})
        return rows
    with handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                errors.append({"file": str(path), "line_number": number, "error": "空行"})
                continue
            try:
                rows.append((number, json.loads(line)))
            except Exception as exc:
                errors.append({"file": str(path), "line_number": number, "error": f"JSON解析失败: {exc}"})
    return rows


def export_sft(raw_path: str | Path, classified_path: str | Path,
               output_path: str | Path = "qwen_sft_train.jsonl",
               errors_path: str | Path = "export_errors.jsonl",
               report_path: str | Path = "export_report.json") -> dict[str, int]:
    errors: list[dict[str, Any]] = []
    raw_rows = _read_jsonl(raw_path, errors, "原始")
    raw_failed = sum(error["line_number"] > 0 for error in errors)
    classified_rows = _read_jsonl(classified_path, errors, "分类")
    roles: dict[str, dict[str, Any]] = {}
    for line_number, row in classified_rows:
        if not isinstance(row, dict) or not isinstance(row.get("id"), str) or not row["id"].strip():
            errors.append({"file": str(classified_path), "line_number": line_number, "error": "分类记录缺少有效id"})
            continue
        roles[row["id"]] = row

    exported_rows: list[dict[str, Any]] = []
    skipped = raw_failed
    for line_number, row in raw_rows:
        if not isinstance(row, dict):
            errors.append({"file": str(raw_path), "line_number": line_number, "error": "原始记录必须是对象"})
            skipped += 1
            continue
        record_id = row.get("id")
        classification = roles.get(record_id) if isinstance(record_id, str) else None
        if classification is None or classification.get("data_role") != "finetune":
            if classification is None:
                errors.append({"file": str(raw_path), "line_number": line_number,
                               "id": record_id, "error": "缺少有效id或对应分类结果"})
            skipped += 1
            continue
        dialogue = row.get("dialogue")
        if not isinstance(dialogue, list) or len(dialogue) < 2:
            errors.append({"file": str(raw_path), "line_number": line_number, "id": record_id,
                           "error": "对话轮数不足，至少需要两轮"})
            skipped += 1
            continue
        messages = []
        invalid = None if isinstance(row.get("source"), str) and row["source"].strip() else "source必须是非空字符串"
        for index, turn in enumerate(dialogue):
            if not isinstance(turn, dict) or not isinstance(turn.get("speaker"), str) or not turn["speaker"].strip() or not isinstance(turn.get("text"), str) or not turn["text"].strip():
                invalid = f"第{index + 1}轮缺少有效text"
                break
            messages.append({"role": "user" if index % 2 == 0 else "assistant", "content": turn["text"]})
        if invalid:
            errors.append({"file": str(raw_path), "line_number": line_number, "id": record_id, "error": invalid})
            skipped += 1
            continue
        tags = classification.get("style_tags", [])
        if not isinstance(tags, list):
            tags = []
        hint = classification.get("community_hint")
        exported_rows.append({"messages": messages,
                              "metadata": {"id": record_id, "source": row.get("source", ""),
                                           "style_tags": tags, "community_hint": hint}})

    Path(output_path).write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in exported_rows), encoding="utf-8")
    Path(errors_path).write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in errors), encoding="utf-8")
    report = {"total_input": len(raw_rows) + raw_failed, "exported": len(exported_rows),
              "skipped": skipped, "error_count": len(errors)}
    Path(report_path).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw")
    parser.add_argument("classified")
    parser.add_argument("--output", default="qwen_sft_train.jsonl")
    parser.add_argument("--errors", default="export_errors.jsonl")
    parser.add_argument("--report", default="export_report.json")
    args = parser.parse_args()
    print(json.dumps(export_sft(args.raw, args.classified, args.output, args.errors, args.report), ensure_ascii=False))


if __name__ == "__main__":
    main()
