#!/usr/bin/env python3
"""将本地 B 站评论 JSON/JSONL 文件接入 QWbot V6 raw_source 格式。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


def _read_records(path: Path) -> Iterable[tuple[int, Any]]:
    """Yield (line number, payload); JSON 文件按一条记录处理。"""
    text = path.read_text(encoding="utf-8")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        for number, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            try:
                yield number, json.loads(line)
            except json.JSONDecodeError as exc:
                yield number, exc
        return
    if isinstance(payload, list):
        for number, item in enumerate(payload, 1):
            yield number, item
    else:
        yield 1, payload


def _convert(payload: Any) -> tuple[dict[str, Any], int]:
    if not isinstance(payload, dict):
        raise ValueError("输入记录必须是对象")
    oid = payload.get("oid")
    if not isinstance(oid, (str, int)) or not str(oid).strip():
        raise ValueError("oid必须存在")
    comments = payload.get("comments")
    if not isinstance(comments, list):
        raise ValueError("comments必须是数组")
    cleaned: list[dict[str, str]] = []
    removed = 0
    for comment in comments:
        if not isinstance(comment, dict):
            removed += 1
            continue
        user, text = comment.get("user"), comment.get("text")
        if not isinstance(user, str) or not user.strip() or not isinstance(text, str) or not text.strip():
            removed += 1
            continue
        cleaned.append({"user": user.strip(), "text": text.strip()})
    if not cleaned:
        raise ValueError("comments清洗后为空")
    title = payload.get("title", "")
    topic_hint = title.strip() if isinstance(title, str) else ""
    return {"source_type": "bilibili_reply_chain", "data": {
        "oid": str(oid).strip(), "topic_hint": topic_hint, "comments": cleaned
    }}, removed


def ingest(input_path: str | Path, output_path: str | Path = "raw_source.jsonl",
           errors_path: str | Path = "ingest_errors.jsonl",
           report_path: str | Path = "bilibili_ingest_report.json") -> dict[str, int]:
    report = {"total_input": 0, "success": 0, "failed": 0, "empty_removed": 0}
    output_path, errors_path, report_path = map(Path, (output_path, errors_path, report_path))
    for path in (output_path, errors_path, report_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output, errors_path.open("w", encoding="utf-8") as errors:
        for line_number, payload in _read_records(Path(input_path)):
            report["total_input"] += 1
            if isinstance(payload, json.JSONDecodeError):
                reason = f"JSON解析失败: {payload.msg}"
                report["failed"] += 1
                errors.write(json.dumps({"line_number": line_number, "error": reason}, ensure_ascii=False) + "\n")
                continue
            try:
                row, removed = _convert(payload)
            except (TypeError, ValueError) as exc:
                report["failed"] += 1
                errors.write(json.dumps({"line_number": line_number, "error": str(exc)}, ensure_ascii=False) + "\n")
                continue
            report["success"] += 1
            report["empty_removed"] += removed
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, default=Path("raw_source.jsonl"))
    parser.add_argument("--errors", type=Path, default=Path("ingest_errors.jsonl"))
    parser.add_argument("--report", type=Path, default=Path("bilibili_ingest_report.json"))
    args = parser.parse_args(argv)
    ingest(args.input, args.output, args.errors, args.report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
