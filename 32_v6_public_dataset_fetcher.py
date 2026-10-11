#!/usr/bin/env python3
"""Fetch a small, auditable first batch of public QWbot V6 data.

The fetcher never fabricates records: a failed remote request is represented in
the report and no placeholder data file is written.
"""
from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _local_size(root: Path) -> int:
    return sum(p.stat().st_size for p in root.rglob("*") if p.is_file()) if root.exists() else 0


def _entry(source: dict[str, Any], *, size: int = 0, status: str = "not_attempted", error: str | None = None) -> dict[str, Any]:
    result = {key: source.get(key, "") for key in ("source", "url", "dataset_type", "format", "suggested_role")}
    result.update({"size": size, "status": status})
    if error:
        result["error"] = error
    return result


def fetch_first(config: Path, output_dir: Path, report: Path, timeout: int = 20) -> list[dict[str, Any]]:
    data = json.loads(config.read_text(encoding="utf-8"))
    sources = data.get("sources", [])
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    attempted = False
    for index, source in enumerate(sources):
        kind = source.get("dataset_type")
        if kind == "local":
            results.append(_entry(source, size=_local_size(Path("datasets")), status="available"))
            continue
        if attempted:
            results.append(_entry(source))
            continue
        attempted = True
        request = urllib.request.Request(source["url"], headers={"User-Agent": "QWbot-V6-public-fetcher/1.0"})
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = response.read()
            # Validate JSON before writing, so an HTML error page is never data.
            parsed = json.loads(payload.decode("utf-8"))
            target = output_dir / f"v6_first_public_{index:02d}.json"
            target.write_text(json.dumps(parsed, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            results.append(_entry(source, size=target.stat().st_size, status="downloaded"))
            break
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError, ValueError) as exc:
            results.append(_entry(source, status="blocked", error=f"{type(exc).__name__}: {exc}"))
            # Keep scanning configuration so the report describes every source,
            # while still attempting at most one remote download per run.
    report_payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": str(config),
        "network_attempted": attempted,
        "sources": results,
    }
    report.write_text(json.dumps(report_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("v6_data_sources.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("datasets/v6_public"))
    parser.add_argument("--report", type=Path, default=Path("data_sources_report.json"))
    parser.add_argument("--timeout", type=int, default=20)
    args = parser.parse_args()
    results = fetch_first(args.config, args.output_dir, args.report, args.timeout)
    for item in results:
        print(f"{item['source']}: {item['status']} ({item['size']} bytes)")


if __name__ == "__main__":
    main()
