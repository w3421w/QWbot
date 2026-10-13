#!/usr/bin/env python3
"""B站采集接口层（当前版本仅生成标准空采集结果，不联网）。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable


def collect_task(task: dict[str, Any]) -> dict[str, Any]:
    """采集单个任务。

    当前实现故意不访问网络，只返回可供真实采集器替换的标准空结果。
    """
    if not isinstance(task, dict):
        raise ValueError("任务必须是对象")
    oid = task.get("oid")
    if not isinstance(oid, (str, int)) or not str(oid).strip():
        raise ValueError("oid必须存在")
    title = task.get("title", "")
    if not isinstance(title, str):
        raise ValueError("title必须是字符串")
    return {"oid": str(oid).strip(), "title": title, "comments": []}


def run_collection_tasks(
    tasks_path: str | Path = "video_tasks.jsonl",
    output_dir: str | Path = "collector_raw",
    report_path: str | Path = "collector_report.json",
    errors_path: str | Path = "collector_errors.jsonl",
    collector: Callable[[dict[str, Any]], dict[str, Any]] = collect_task,
) -> dict[str, int]:
    """批量执行任务并写入原始结果、报告和错误记录。

    ``collector`` 是未来真实 B 站采集器的替换接口。
    """
    tasks_path, output_dir = Path(tasks_path), Path(output_dir)
    report_path, errors_path = Path(report_path), Path(errors_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    errors_path.parent.mkdir(parents=True, exist_ok=True)
    report = {"total_tasks": 0, "success": 0, "failed": 0}
    with errors_path.open("w", encoding="utf-8") as errors:
        try:
            lines = tasks_path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeError) as exc:
            lines = []
            report["failed"] = 1
            errors.write(json.dumps({"line": 0, "error": str(exc)}, ensure_ascii=False) + "\n")
        for line_number, line in enumerate(lines, 1):
            if not line.strip():
                continue
            report["total_tasks"] += 1
            try:
                task = json.loads(line)
                result = collector(task)
                oid = str(result["oid"]).strip()
                if not oid or Path(oid).name != oid:
                    raise ValueError("结果oid不可用于文件名")
                (output_dir / f"{oid}.json").write_text(
                    json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
                )
            except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError, KeyError) as exc:
                report["failed"] += 1
                errors.write(json.dumps({"line": line_number, "error": str(exc)}, ensure_ascii=False) + "\n")
            else:
                report["success"] += 1
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", type=Path, default=Path("video_tasks.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("collector_raw"))
    parser.add_argument("--report", type=Path, default=Path("collector_report.json"))
    parser.add_argument("--errors", type=Path, default=Path("collector_errors.jsonl"))
    args = parser.parse_args(argv)
    run_collection_tasks(args.input, args.output_dir, args.report, args.errors)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
