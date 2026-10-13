#!/usr/bin/env python3
"""QWbot V6 JSONL 批量来源适配器（只做本地转换，不联网或下载）。"""
from __future__ import annotations

import argparse
import importlib.util
import json
from collections import Counter
from pathlib import Path
from typing import Any

_ADAPTER_PATH = Path(__file__).with_name("18_v6_source_adapter.py")
_spec = importlib.util.spec_from_file_location("v6_source_adapter", _ADAPTER_PATH)
_adapter = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_adapter)


def _error_text(exc: Exception) -> str:
    message = str(exc).strip()
    return message or exc.__class__.__name__


def convert_jsonl(input_path: str | Path, output_path: str | Path,
                  errors_path: str | Path, report_path: str | Path) -> dict[str, Any]:
    """转换一个 raw_source.jsonl，并返回写入的统计报告。"""
    input_path, output_path = Path(input_path), Path(output_path)
    errors_path, report_path = Path(errors_path), Path(report_path)
    total = success = failed = 0
    source_counts: Counter[str] = Counter()

    with input_path.open("r", encoding="utf-8-sig") as source, \
            output_path.open("w", encoding="utf-8") as output, \
            errors_path.open("w", encoding="utf-8") as errors:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            total += 1
            source_type: Any = None
            try:
                raw = json.loads(line)
                if not isinstance(raw, dict):
                    raise ValueError("原始记录必须是对象")
                source_type = raw.get("source_type")
                if isinstance(source_type, str):
                    source_counts[source_type] += 1
                payload = raw.get("data")
                if "source_type" not in raw or "data" not in raw:
                    raise ValueError("原始记录必须包含source_type和data")
                row = _adapter.adapt_source(source_type, payload)
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
                success += 1
            except Exception as exc:  # 单条错误不得中断批次
                failed += 1
                errors.write(json.dumps({"line_number": line_number,
                                         "error": _error_text(exc),
                                         "source_type": source_type},
                                        ensure_ascii=False) + "\n")

    report = {"total": total, "success": success, "failed": failed,
              "source_counts": dict(source_counts)}
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="raw_source.jsonl")
    parser.add_argument("--output", default="v6_dialogue_raw.jsonl")
    parser.add_argument("--errors", default="adapter_errors.jsonl")
    parser.add_argument("--report", default="adapter_report.json")
    args = parser.parse_args()
    report = convert_jsonl(args.input, args.output, args.errors, args.report)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
