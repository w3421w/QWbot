#!/usr/bin/env python3
"""从本地视频 JSONL 生成 B 站采集任务、错误记录及统计报告，不联网。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def build_task(record: Any) -> dict[str, Any]:
    """校验记录并提取标准任务字段；可选字段缺失时使用空值。"""
    if not isinstance(record, dict):
        raise ValueError("记录必须是对象")
    oid = record.get("oid")
    if isinstance(oid, bool) or not isinstance(oid, (str, int)) or not str(oid).strip():
        raise ValueError("oid必须存在且为非空字符串或整数")
    title = record.get("title", "")
    category = record.get("category", "")
    tags = record.get("tags", [])
    if not isinstance(title, str):
        raise ValueError("title必须是字符串")
    if not isinstance(category, str):
        raise ValueError("category必须是字符串")
    if not isinstance(tags, list) or any(not isinstance(tag, str) for tag in tags):
        raise ValueError("tags必须是字符串列表")
    return {"oid": str(oid).strip(), "title": title, "category": category, "tags": tags}


def build_video_tasks(
    input_path: str | Path = "local_video_list.jsonl",
    output_path: str | Path = "video_tasks.jsonl",
    report_path: str | Path = "task_report.json",
    errors_path: str | Path = "task_errors.jsonl",
) -> dict[str, Any]:
    """逐行生成任务，保留每个 oid 的首条有效记录。

    total_input 统计非空行（包括无效 JSON）；分类统计仅包含去重后的任务。
    输入读取失败会向调用者抛出异常，避免生成误导性的成功报告。
    """
    input_path, output_path = Path(input_path), Path(output_path)
    report_path, errors_path = Path(report_path), Path(errors_path)
    paths = [input_path, output_path, report_path, errors_path]
    if len({path.resolve() for path in paths}) != len(paths):
        raise ValueError("输入、任务、报告和错误文件路径必须互不相同")
    report: dict[str, Any] = {
        "total_input": 0, "valid_tasks": 0, "duplicate_count": 0,
        "invalid_count": 0, "category_distribution": {},
    }
    seen: set[str] = set()
    with input_path.open(encoding="utf-8") as source:
        for path in (output_path, report_path, errors_path):
            path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as tasks, errors_path.open("w", encoding="utf-8") as errors:
            for line_number, line in enumerate(source, 1):
                if not line.strip():
                    continue
                report["total_input"] += 1
                try:
                    task = build_task(json.loads(line))
                except (ValueError, TypeError) as exc:
                    report["invalid_count"] += 1
                    errors.write(json.dumps({"line": line_number, "error": str(exc)}, ensure_ascii=False) + "\n")
                    continue
                if task["oid"] in seen:
                    report["duplicate_count"] += 1
                    continue
                seen.add(task["oid"])
                tasks.write(json.dumps(task, ensure_ascii=False) + "\n")
                report["valid_tasks"] += 1
                distribution = report["category_distribution"]
                distribution[task["category"]] = distribution.get(task["category"], 0) + 1
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", type=Path, default=Path("local_video_list.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("video_tasks.jsonl"))
    parser.add_argument("--report", type=Path, default=Path("task_report.json"))
    parser.add_argument("--errors", type=Path, default=Path("task_errors.jsonl"))
    args = parser.parse_args(argv)
    build_video_tasks(args.input, args.output, args.report, args.errors)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
