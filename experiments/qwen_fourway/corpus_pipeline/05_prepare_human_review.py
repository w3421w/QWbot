#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prepare a fixed-seed, stratified manual-review sample.

This script only reads the V1 candidate JSONL and raw source files.  It writes
an independent UTF-8-SIG CSV and a sampling report; it never creates training
data and never fills any reviewer decision fields.
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline_common import ROOT, RAW_DIR, csv_write, norm


MANUAL_FIELDS = [
    "text_quality", "context_sufficiency", "reply_target_verifiable",
    "style_value", "hallucination_risk", "pii_status", "reviewer_decision",
    "reviewer_notes",
]
OUTPUT_FIELDS = [
    "id", "source", "conversation_id", "source_post_id", "source_reply_id",
    "source_file", "reply_index", "post_title", "op_body", "original_reply",
    "target_reply", "context", "incoming_message", "reply_relation",
    "quality_flags", "quality_score", "provenance_type", "review_status",
    "license_status", "text_hash", "near_duplicate_of", "sampling_strata",
    "sampling_seed", "sampling_method",
] + MANUAL_FIELDS


def load_candidates(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict) and row.get("review_status") == "pending_review":
                row["_line_no"] = line_no
                rows.append(row)
    return rows


def load_raw_context(rows: list[dict]) -> dict[tuple[str, str], tuple[str, str, str]]:
    """Return (source_file, post_id) -> (title, body, raw reply by index later)."""
    wanted = {(r.get("source_file", ""), str(r.get("source_post_id", ""))) for r in rows}
    out = {}
    for source_file, pid in sorted(wanted):
        if not source_file:
            continue
        path = ROOT / source_file
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for ix, item in enumerate(data if isinstance(data, list) else []):
            if not isinstance(item, dict):
                continue
            link = str(item.get("链接", ""))
            m = re.search(r"/p/(\d+)", link)
            item_pid = m.group(1) if m else ""
            if item_pid == pid:
                out[(source_file, pid)] = (
                    norm(item.get("标题")), norm(item.get("楼主内容")),
                    json.dumps(item.get("回复列表", []), ensure_ascii=False),
                )
                break
    return out


def raw_fields(row: dict, raw_lookup: dict) -> tuple[str, str, str]:
    key = (row.get("source_file", ""), str(row.get("source_post_id", "")))
    title, body, replies_json = raw_lookup.get(key, ("", "", "[]"))
    try:
        replies = json.loads(replies_json)
    except json.JSONDecodeError:
        replies = []
    ix = row.get("reply_index", -1)
    try:
        original = norm(replies[int(ix)]) if 0 <= int(ix) < len(replies) else ""
    except (TypeError, ValueError):
        original = ""
    return title, body, original


def strata(row: dict, title: str, body: str, original: str) -> list[str]:
    text = original or row.get("target_reply", "") or ""
    flags = row.get("quality_flags", "") or ""
    has_body = bool(body and body != "[无法获取楼主内容]")
    if len(text) <= 15:
        length = "short"
    elif len(text) >= 120:
        length = "long"
    else:
        length = "medium"
    tags = ["body_present" if has_body else "body_absent", length]
    if "exact_duplicate" in flags or "near_duplicate_risk" in flags:
        tags.append("duplicate_or_near_duplicate")
    else:
        tags.append("non_duplicate")
    if "privacy_review" in flags or re.search(r"(?:1[3-9]\d{9}|微信|微 信|QQ号|邮箱|@)", text, re.I):
        tags.append("privacy_risk")
    else:
        tags.append("no_privacy_flag")
    return tags


def choose(rows: list[dict], enriched: list[dict], size: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    # Shuffle each pool once with the fixed seed.  Mandatory risk/duplicate
    # rows are included first, followed by post coverage, then balanced pools.
    pools = collections.defaultdict(list)
    for row in enriched:
        for tag in row["_strata"]:
            pools[tag].append(row)
    for pool in pools.values():
        rng.shuffle(pool)
    selected, seen = [], set()

    def add(row):
        if len(selected) < size and row["id"] not in seen:
            seen.add(row["id"]); selected.append(row)

    for tag in ("privacy_risk", "duplicate_or_near_duplicate"):
        for row in pools[tag]:
            add(row)
    by_post = collections.defaultdict(list)
    for row in enriched:
        by_post[row.get("source_post_id", "")].append(row)
    post_order = list(by_post)
    rng.shuffle(post_order)
    for pid in post_order:
        if by_post[pid]:
            add(rng.choice(by_post[pid]))
    core = ["body_present", "body_absent", "short", "long", "medium"]
    while len(selected) < size:
        progressed = False
        for tag in core:
            for row in pools[tag]:
                if row["id"] not in seen:
                    add(row); progressed = True; break
            if len(selected) >= size:
                break
        if not progressed:
            break
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a fixed-seed stratified V1.1 manual-review CSV; no training data is produced.")
    parser.add_argument("--input-candidates", type=Path, default=Path("experiments/qwen_fourway/corpus_pipeline/output_sample2/clean_candidates.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/qwen_fourway/corpus_pipeline/output_human_review_v11"))
    parser.add_argument("--sample-size", type=int, default=150)
    parser.add_argument("--seed", type=int, default=1101)
    parser.add_argument("--check-only", action="store_true", help="Validate and report candidate count without writing output.")
    args = parser.parse_args()
    if args.sample_size <= 0:
        raise SystemExit("--sample-size must be positive")
    if not args.input_candidates.exists():
        raise SystemExit(f"missing candidate file: {args.input_candidates}")
    rows = load_candidates(args.input_candidates)
    if not rows:
        raise SystemExit("no pending_review candidates found")
    if args.check_only:
        print(json.dumps({"pending_candidates": len(rows), "requested_sample_size": args.sample_size}, ensure_ascii=False, indent=2)); return
    raw_lookup = load_raw_context(rows)
    enriched = []
    for row in rows:
        title, body, original = raw_fields(row, raw_lookup)
        tags = strata(row, title, body, original)
        out = dict(row)
        out.update({"post_title": title, "op_body": body, "original_reply": original,
                    "_strata": tags})
        enriched.append(out)
    selected = choose(rows, enriched, min(args.sample_size, len(enriched)), args.seed)
    for row in selected:
        row["sampling_strata"] = ",".join(row.pop("_strata"))
        row["sampling_seed"] = str(args.seed)
        row["sampling_method"] = "fixed_seed_stratified; risk_duplicate_first_post_coverage_round_robin"
        for field in MANUAL_FIELDS:
            row[field] = ""
        row.pop("_line_no", None)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "human_review_sample_v11.csv"
    csv_write(csv_path, selected, OUTPUT_FIELDS)
    counts = collections.Counter()
    for row in selected:
        counts.update(x for x in row["sampling_strata"].split(",") if x)
    report = {
        "input_candidates": len(rows), "sampled": len(selected), "requested_sample_size": args.sample_size,
        "seed": args.seed, "raw_context_recovered": sum(bool(r["post_title"] or r["op_body"] or r["original_reply"]) for r in selected),
        "unique_source_posts": len({r.get("source_post_id") for r in selected}),
        "strata_counts": dict(counts), "csv": str(csv_path),
        "manual_fields_blank": MANUAL_FIELDS,
        "relation_policy": "reply_target_verifiable remains blank; source relation is unknown and must be manually assessed",
        "authorization_policy": "accept does not establish source authorization",
    }
    (args.output_dir / "sampling_report_v11.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
