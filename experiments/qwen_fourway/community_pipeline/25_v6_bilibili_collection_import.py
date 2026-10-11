#!/usr/bin/env python3
"""将本地收集的 B 站原始 JSON 文件导入 QWbot V6 collection 格式。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _convert(payload: Any) -> tuple[dict[str, Any], int]:
    if not isinstance(payload, dict):
        raise ValueError("输入记录必须是对象")
    oid = payload.get("oid")
    if not isinstance(oid, (str, int)) or not str(oid).strip():
        raise ValueError("oid必须存在")
    comments = payload.get("comments")
    if not isinstance(comments, list):
        raise ValueError("comments必须是数组")
    cleaned: list[dict[str, Any]] = []
    removed = 0
    for comment in comments:
        if not isinstance(comment, dict):
            removed += 1
            continue
        user, text = comment.get("user"), comment.get("text")
        if not isinstance(user, str) or not user.strip() or not isinstance(text, str) or not text.strip():
            removed += 1
            continue
        item: dict[str, Any] = {"user": user.strip(), "text": text.strip()}
        if "reply_to" in comment:
            item["reply_to"] = comment["reply_to"]
        cleaned.append(item)
    if not cleaned:
        raise ValueError("comments清洗后为空")
    title = payload.get("title", "")
    topic_hint = title.strip() if isinstance(title, str) else ""
    return {
        "source_type": "bilibili_reply_chain",
        "data": {"oid": str(oid).strip(), "topic_hint": topic_hint, "comments": cleaned},
    }, removed


def import_collection(
    input_dir: str | Path,
    output_path: str | Path = "bilibili_raw_collection.jsonl",
    errors_path: str | Path = "collection_errors.jsonl",
    report_path: str | Path = "collection_report.json",
) -> dict[str, int]:
    """Import every ``*.json`` file below *input_dir* and write collection artifacts."""
    source = Path(input_dir)
    output_path, errors_path, report_path = map(Path, (output_path, errors_path, report_path))
    for path in (output_path, errors_path, report_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    files = sorted(source.rglob("*.json")) if source.is_dir() else []
    report = {"files_total": len(files), "files_success": 0, "files_failed": 0, "comments_total": 0}
    with output_path.open("w", encoding="utf-8") as output, errors_path.open("w", encoding="utf-8") as errors:
        for path in files:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                row, _removed = _convert(payload)
            except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
                report["files_failed"] += 1
                errors.write(json.dumps({"file": str(path), "error": str(exc)}, ensure_ascii=False) + "\n")
                continue
            report["files_success"] += 1
            report["comments_total"] += len(row["data"]["comments"])
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, default=Path("bilibili_raw_collection.jsonl"))
    parser.add_argument("--errors", type=Path, default=Path("collection_errors.jsonl"))
    parser.add_argument("--report", type=Path, default=Path("collection_report.json"))
    args = parser.parse_args(argv)
    import_collection(args.input, args.output, args.errors, args.report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
