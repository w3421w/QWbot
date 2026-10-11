#!/usr/bin/env python3
"""QWbot V5: external meme source discovery and review candidates."""
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from collections import Counter
from datetime import datetime
from pathlib import Path

P = Path(__file__).resolve().parent
sys.path.insert(0, str(P))

from common.core import privacy
from common.classifier import classify

OUT = P / (
    "output_v5_external_"
    + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
)

SOURCES = {
    "chime": "https://github.com/yuboxie/chime",
    "meme_bench": (
        "https://huggingface.co/datasets/"
        "IndexTeam/Meme-Translation-Bench"
    )
}


def clean(value, flags):
    if isinstance(value, str):
        safe, found = privacy(value)
        flags.update(found)
        return safe
    if isinstance(value, list):
        return [clean(v, flags) for v in value]
    if isinstance(value, dict):
        return {k: clean(v, flags) for k, v in value.items()}
    return value


def categorize(text):
    result = classify(text[:4000])
    return result.get("topics") or ["unknown"]


def fetch_bench():
    dataset = "IndexTeam/Meme-Translation-Bench"
    output = []
    failures = []

    # Spread across the dataset; no full dataset download.
    for offset in range(0, 3600, 300):
        query = urllib.parse.urlencode({
            "dataset": dataset,
            "config": "default",
            "split": "test",
            "offset": offset,
            "length": 20
        })

        url = (
            "https://datasets-server.huggingface.co/rows?"
            + query
        )

        req = urllib.request.Request(
            url,
            headers={
                "Accept": "application/json",
                "User-Agent": "QWbot-V5-Research-Pilot"
            }
        )

        try:
            with urllib.request.urlopen(
                req, timeout=20
            ) as response:
                raw = response.read(2_000_001)

            if len(raw) > 2_000_000:
                raise ValueError("Response too large")

            obj = json.loads(raw)
            rows = obj.get("rows", [])

            for item in rows:
                row = item.get("row")

                if not isinstance(row, dict):
                    continue

                flags = set()

                fields = clean({
                    "term": row.get("term"),
                    "original_text": row.get("original_text"),
                    "definition": row.get("definition"),
                    "video_title": row.get("video_title"),
                    "text_type": row.get("text_type"),
                    "examples": row.get("examples", [])
                }, flags)

                topic_text = " ".join(str(fields.get(k) or "")
                    for k in ("term", "definition", "video_title"))

                output.append({
                    "source": dataset,
                    "source_row_index": item.get("row_idx"),
                    "fields": fields,
                    "topics": categorize(topic_text),
                    "privacy_flags": sorted(flags),
                    "review_status": (
                        "privacy_quarantine"
                        if {"ambiguous_mention", "private_identity"} & flags
                        else "unreviewed"
                    ),
                    "data_role": "benchmark_reference_only",
                    "license_status": "cc-by-nc-4.0_underlying_rights_unverified",
                    "trainable": False
                })

            print(
                f"MEME_BENCH offset={offset}: "
                f"{len(rows)} rows"
            )

        except Exception as exc:
            failures.append({
                "offset": offset,
                "error": str(exc)
            })
            print("SOURCE_ERROR:", offset, exc)

        time.sleep(1)

    return output, failures


def load_chime():
    path = (
        P / "external_source_index/chime/data/chime_full.json"
    )

    if not path.is_file():
        return [], ["CHIME file not found"]

    data = json.loads(path.read_text(encoding="utf-8-sig"))

    if isinstance(data, dict):
        for key in ("data", "memes", "records", "items"):
            if isinstance(data.get(key), list):
                data = data[key]
                break

    if not isinstance(data, list):
        return [], ["Unexpected CHIME JSON structure"]

    output = []

    for index, item in enumerate(data):
        if not isinstance(item, dict):
            continue

        flags = set()
        fields = clean({
            k: item.get(k)
            for k in (
                "meme", "meaning", "origin",
                "examples", "type_cn", "type_en",
                "profanity", "offense"
            )
        }, flags)

        topic_text = " ".join(str(fields.get(k) or "")
            for k in ("meme", "meaning", "origin"))

        output.append({
            "source": "yuboxie/chime",
            "source_row_index": index,
            "fields": fields,
            "topics": categorize(topic_text),
            "privacy_flags": sorted(flags),
            "review_status": (
                "privacy_quarantine"
                if {"ambiguous_mention", "private_identity"} & flags
                else "unreviewed"
            ),
            "data_role": "meme_reference_only",
            "license_status": "repository_mit_underlying_rights_review",
            "trainable": False
        })

    return output, []


def write_new(path, records):
    with path.open("x", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.chmod(path, 0o600)


def main():
    chime, chime_errors = load_chime()
    bench, bench_errors = fetch_bench()

    OUT.mkdir(mode=0o700, parents=True, exist_ok=False)

    write_new(OUT / "chime_review.jsonl", chime)
    write_new(OUT / "meme_bench_review.jsonl", bench)

    counts = Counter(
        topic
        for r in (chime + bench)
        for topic in set(r["topics"])
    )

    report = {
        "sources": SOURCES,
        "chime_records": len(chime),
        "meme_bench_sample_records": len(bench),
        "topic_counts": dict(counts.most_common()),
        "errors": chime_errors + bench_errors,
        "trainable": False,
        "use": "source discovery and manual review only"
    }

    with (OUT / "manifest.json").open("x", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("OUTPUT:", OUT)


if __name__ == "__main__":
    main()
