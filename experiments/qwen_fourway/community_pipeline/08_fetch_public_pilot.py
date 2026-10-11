#!/usr/bin/env python3
"""Inspect/download bounded Hugging Face public pilot samples.

Default mode fetches dataset metadata only.  ``--download-pilot`` uses the
Hugging Face datasets-server first-rows API and never downloads a repository
file or a complete dataset.  This module deliberately has no HF token
support; public access control is respected by urllib.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from common.core import privacy, write_new

MAX_RESPONSE_BYTES = 2 * 1024 * 1024
DEFAULT_TIMEOUT = 15
HF_API = "https://huggingface.co/api/datasets/"
ROWS_API = "https://datasets-server.huggingface.co/first-rows"
SPLITS_API = "https://datasets-server.huggingface.co/splits"
SOURCES = {
    "Midsummra/bilibilicomment": {
        "kind": "bilibili_independent_comment",
        "fields": ["message", "time", "timestamp"],
        "declared_license": "unknown",
        "rights_note": "Dataset declaration must be checked separately from rights of underlying Bilibili comments; no parent/comment ID confirmed.",
    },
    "helloollel/chat_dataset_bilibili": {
        "kind": "derived_instruction_response",
        "fields": ["instruction", "input", "output"],
        "declared_license": "unknown",
        "rights_note": "Instruction/output pairs are not assumed to be real user-to-user reply chains; underlying text rights require separate review.",
    },
}


def _json_request(url: str, timeout: int = DEFAULT_TIMEOUT) -> tuple[dict, bytes]:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https" or parsed.hostname not in {"huggingface.co", "datasets-server.huggingface.co"}:
        raise RuntimeError(f"refusing non-Hugging-Face URL: {url}")
    req = urllib.request.Request(url, headers={"Accept": "application/json", "User-Agent": "QWbot-community-pilot/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            chunks, total = [], 0
            while True:
                chunk = response.read(min(65536, MAX_RESPONSE_BYTES + 1 - total))
                if not chunk:
                    break
                chunks.append(chunk); total += len(chunk)
                if total > MAX_RESPONSE_BYTES:
                    raise RuntimeError(f"response exceeds {MAX_RESPONSE_BYTES} bytes: {url}")
            raw = b"".join(chunks)
            if response.status < 200 or response.status >= 300:
                raise RuntimeError(f"HTTP {response.status} from {url}")
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"HTTP {exc.code} from {url}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"request failed for {url}: {exc.reason}") from exc
    except TimeoutError as exc:
        raise RuntimeError(f"request timed out for {url}") from exc
    try:
        return json.loads(raw.decode("utf-8")), raw
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid JSON response from {url}") from exc


def inspect_source(repo: str, timeout: int = DEFAULT_TIMEOUT) -> dict:
    if repo not in SOURCES:
        raise ValueError(f"source is not allowlisted: {repo}")
    url = HF_API + urllib.parse.quote(repo, safe="/")
    data, raw = _json_request(url, timeout)
    card = data.get("cardData") if isinstance(data.get("cardData"), dict) else {}
    siblings = data.get("siblings") if isinstance(data.get("siblings"), list) else []
    declared = card.get("license") or data.get("license") or SOURCES[repo]["declared_license"]
    return {
        "dataset": repo,
        "metadata_url": url,
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "private": bool(data.get("private", False)),
        "gated": bool(data.get("gated", False)),
        "downloads": data.get("downloads"),
        "likes": data.get("likes"),
        "last_modified": data.get("lastModified"),
        "declared_license": declared,
        "license_source": "dataset_card_or_api" if declared != "unknown" else "not_declared_in_metadata",
        "rights_audit": SOURCES[repo]["rights_note"],
        "expected_fields": SOURCES[repo]["fields"],
        "siblings_count": len(siblings),
        "access": "public_metadata_observed",
    }


def fetch_rows(repo: str, limit: int, timeout: int = DEFAULT_TIMEOUT) -> tuple[list[dict], str, bytes]:
    """Fetch a bounded, nonrandom first-rows preview from a discovered config/split."""
    if repo not in SOURCES:
        raise ValueError(f"source is not allowlisted: {repo}")
    if not 1 <= limit <= 100:
        raise ValueError("limit must be between 1 and 100")

    splits_url = SPLITS_API + "?" + urllib.parse.urlencode({"dataset": repo})
    splits_response, _ = _json_request(splits_url, timeout)
    available = splits_response.get("splits")
    if not isinstance(available, list):
        raise RuntimeError(f"dataset splits not available: {splits_url}")
    choices = [entry for entry in available
               if isinstance(entry, dict)
               and isinstance(entry.get("config"), str) and entry["config"]
               and isinstance(entry.get("split"), str) and entry["split"]
               and entry.get("dataset", repo) == repo]
    if not choices:
        raise RuntimeError(f"no usable config/split found: {splits_url}")
    # Prefer train and default *when present*. Never assume either exists.
    chosen = sorted(choices, key=lambda e: (
        e["split"] != "train", e["config"] != "default", e["config"], e["split"]
    ))[0]
    config, split = chosen["config"], chosen["split"]
    query = urllib.parse.urlencode({"dataset": repo, "config": config, "split": split})
    url = ROWS_API + "?" + query
    data, raw = _json_request(url, timeout)
    rows = data.get("rows")
    if not isinstance(rows, list):
        raise RuntimeError(f"sample response has no rows list: {url}")
    out = []
    for idx, item in enumerate(rows[:limit]):
        if isinstance(item, dict) and isinstance(item.get("row"), dict):
            record = dict(item["row"])
            row_idx = item.get("row_idx")
            record["__qwb_preview_idx"] = row_idx if isinstance(row_idx, int) and row_idx >= 0 else idx
            record["__qwb_preview_config"] = config
            record["__qwb_preview_split"] = split
            out.append(record)
    return out, url, raw


def _base_record(repo: str, row: dict, index: int, source_file: str, declared_license: str) -> dict:
    original = row.get("message", "") if repo.startswith("Midsummra/") else row.get("output", "")
    safe_text, pflags = privacy(original if isinstance(original, str) else "")
    preview_idx = row.get("__qwb_preview_idx", index)
    if not isinstance(preview_idx, int) or preview_idx < 0:
        preview_idx = index
    config = row.get("__qwb_preview_config")
    split = row.get("__qwb_preview_split")
    record_key = f"{config}:{split}:{preview_idx}" if isinstance(config, str) and isinstance(split, str) else str(preview_idx)
    created = row.get("time", row.get("timestamp"))
    if isinstance(created, str):
        created = privacy(created)[0]
    # Missing conversation participants and reply edges remain explicitly unknown.
    return {
        "platform": "bilibili", "domain": [], "topics": [], "subtopics": [],
        "classification_confidence": 0.0, "classification_evidence": [], "classification_version": None,
        "thread_id": None, "message_id": f"public:{repo}:{record_key}",
        "parent_comment_id": None, "root_comment_id": None, "speaker_id": None,
        "message_order": None, "text": safe_text, "content_type": "comment",
        "created_at": created, "video_id": None, "post_id": None,
        "source_url": f"https://huggingface.co/datasets/{repo}", "source_file": str(Path(source_file).resolve()),
        "source_record_id": record_key, "provenance_type": "public_dataset_sample", "is_synthetic": False,
        "source_license": declared_license, "license_status": "dataset_declared_unverified" if declared_license != "unknown" else "verification_pending",
        "review_status": "unreviewed", "trainable": False,
        "quality_flags": ["no_confirmed_thread_id", "no_confirmed_parent_id", "no_confirmed_speaker_id", "first_rows_not_random"],
        "privacy_flags": pflags, "parse_status": "parsed_public_adapter", "split": None, "source_group_id": None,
        "public_dataset": repo,
    }


def adapt_rows(repo: str, rows: list[dict], source_file: str, declared_license: str = "unknown") -> list[dict]:
    if repo not in SOURCES:
        raise ValueError(f"source is not allowlisted: {repo}")
    records = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        if repo == "Midsummra/bilibilicomment":
            if not isinstance(row.get("message"), str) or not row["message"].strip():
                continue
        elif not isinstance(row.get("output"), str) or not row["output"].strip():
            continue
        record = _base_record(repo, row, index, source_file, declared_license)
        # Privacy flags from all exported text fields (not only record['text']).
        all_flags = set(record["privacy_flags"])
        def protected(value):
            """Sanitize every text value, including nested JSON containers."""
            if isinstance(value, str):
                safe, flags = privacy(value)
                all_flags.update(flags)
                return safe
            if isinstance(value, list):
                return [protected(item) for item in value]
            if isinstance(value, dict):
                return {key: protected(item) for key, item in value.items()}
            return value
        record["created_at"] = protected(record["created_at"])
        if repo == "Midsummra/bilibilicomment":
            record["source_fields"] = {k: protected(row.get(k)) for k in ("message", "time", "timestamp")}
            record["adapter_kind"] = "bilibili_independent_comment_candidate"
        else:
            for key in ("instruction", "input", "output"):
                record[key] = protected(row.get(key))
            record["relation_type"] = "derived_instruction_response"
            record["adapter_kind"] = "derived_instruction_response"
            record["quality_flags"].append("not_verified_multi_user_chat")
        record["privacy_flags"] = sorted(all_flags)
        records.append(record)
    return records


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Check HF metadata or fetch <=100 public pilot rows (no token, no full dataset)")
    p.add_argument("--source", choices=["all", *SOURCES], default="all")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--download-pilot", action="store_true", help="also call datasets-server first-rows")
    p.add_argument("--limit", type=int, default=100, help="sample rows per source, maximum 100")
    p.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    return p


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    if not 1 <= args.limit <= 100:
        raise SystemExit("error: --limit must be between 1 and 100")
    repos = list(SOURCES) if args.source == "all" else [args.source]
    outdir = Path(args.output_dir); outdir.mkdir(parents=True, exist_ok=True)
    for repo in repos:
        safe = repo.replace("/", "__")
        try:
            metadata = inspect_source(repo, args.timeout)
            write_new(outdir / f"{safe}.metadata.json", metadata)
            if args.download_pilot:
                rows, sample_url, raw = fetch_rows(repo, args.limit, args.timeout)
                records = adapt_rows(repo, rows, str(outdir / f"{safe}.pilot.jsonl"), metadata["declared_license"])
                write_new(outdir / f"{safe}.pilot.jsonl", records, jsonl=True)
                manifest = {"dataset": repo, "sample_url": sample_url, "sample_response_sha256": hashlib.sha256(raw).hexdigest(),
                            "requested_limit": args.limit, "retained_records": len(records), "max_per_source": 100,
                            "dataset_metadata_sha256": metadata["sha256"], "trainable": False,
                            "license_audit": {"dataset_declaration": metadata["declared_license"], "underlying_rights": SOURCES[repo]["rights_note"], "automatic_training_clearance": False}}
                write_new(outdir / f"{safe}.manifest.json", manifest)
            print(f"{repo}: metadata checked" + ("; pilot saved" if args.download_pilot else ""))
        except (RuntimeError, ValueError, FileExistsError) as exc:
            raise SystemExit(f"error [{repo}]: {exc}") from exc
    return 0


if __name__ == "__main__":
    sys.exit(main())
