#!/usr/bin/env python3
"""QWbot V5: offline import of reconstructed public dialogue paths."""

import argparse
import csv
import hashlib
import hmac
import json
import os
import secrets
from pathlib import Path

from common.core import privacy
from common.classifier import classify


def get_key():
    path = Path.home() / ".config/qwbot/v5_hmac.key"
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)

    if not path.exists():
        try:
            fd = os.open(
                str(path),
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                0o600
            )
        except FileExistsError:
            pass
        else:
            with os.fdopen(fd, "wb") as f:
                f.write(secrets.token_bytes(32))

    if path.stat().st_mode & 0o077:
        raise RuntimeError("Unsafe anonymization key permissions")

    key = path.read_bytes()

    if len(key) != 32:
        raise RuntimeError("Invalid anonymization key")

    return key


def anon(key, category, value):
    value = str(value).encode("utf-8")
    digest = hmac.new(
        key, value, hashlib.sha256
    ).hexdigest()[:24]

    return f"{category}:{digest}"


def protect_csv(value):
    value = "" if value is None else str(value)

    if value.lstrip(" \t\r\n\ufeff")[:1] in ("=", "+", "-", "@"):
        return "'" + value

    return value


def write_jsonl(path, rows):
    with path.open("x", encoding="utf-8") as f:
        for row in rows:
            f.write(
                json.dumps(row, ensure_ascii=False) + "\n"
            )

    os.chmod(path, 0o600)


def run(input_dir, output_dir):
    src = Path(input_dir).resolve()
    out = Path(output_dir).resolve()

    if out.exists():
        raise FileExistsError(
            f"Refusing to overwrite: {out}"
        )

    files = sorted(src.rglob("*_convs.json"))

    if not files:
        raise RuntimeError("No conversation JSON files found")

    if len(files) > 100:
        raise RuntimeError("Unexpected number of input files")

    key = get_key()

    conversations = []
    messages = []
    quarantine = []
    review = []
    source_info = []

    seen = set()
    rejected = 0

    for file in files:
        if not file.is_file() or file.stat().st_size > 2_000_000:
            raise RuntimeError("Invalid source file")

        binary = file.read_bytes()
        file_hash = hashlib.sha256(binary).hexdigest()
        data = json.loads(binary.decode("utf-8-sig"))

        if not isinstance(data, list):
            raise ValueError("Unexpected JSON structure")

        group = anon(
            key,
            "source_group",
            file.relative_to(src).as_posix()
        )

        source_info.append({
            "source_group_id": group,
            "sha256": file_hash,
            "paths": len(data)
        })

        for path_index, path in enumerate(data):
            if len(conversations) >= 4000:
                raise RuntimeError("Path count safety limit reached")

            if not isinstance(path, list) or not 2 <= len(path) <= 100:
                rejected += 1
                continue

            turns = []
            privacy_flags = set()

            for idx, item in enumerate(path):
                if not isinstance(item, dict):
                    turns = []
                    break

                speaker = item.get("from")
                value = item.get("value")

                if (
                    not isinstance(speaker, str)
                    or not speaker.strip()
                    or not isinstance(value, str)
                    or not value.strip()
                ):
                    turns = []
                    break

                text, flags = privacy(value.strip()[:2000])
                privacy_flags.update(flags)

                turns.append({
                    "message_order": idx,
                    "speaker_id": anon(key, "speaker", speaker),
                    "text": text,
                    "privacy_flags": flags
                })

            if not turns:
                rejected += 1
                continue

            fingerprint = hashlib.sha256(
                json.dumps(
                    [
                        (t["speaker_id"], t["text"])
                        for t in turns
                    ],
                    ensure_ascii=False
                ).encode()
            ).hexdigest()

            if fingerprint in seen:
                rejected += 1
                continue

            seen.add(fingerprint)

            cid = anon(
                key, "conversation",
                f"{group}:{path_index}"
            )

            isolated = bool(
                {"ambiguous_mention", "private_identity"}
                & privacy_flags
            )

            status = (
                "privacy_quarantine"
                if isolated else "unreviewed"
            )

            topic_result = classify(
                " ".join(t["text"] for t in turns)[:4000]
            )

            conv = {
                "conversation_id": cid,
                "source_group_id": group,
                "source_file_sha256": file_hash,

                "conversation_kind": "reconstructed_branch_path",
                "relation_verification": "unverified",

                "turns": turns,
                "topics": topic_result.get("topics", ["unknown"]),
                "domain": topic_result.get("domain", []),

                "quality_flags": [
                    "reconstructed_branch_path",
                    "no_verified_reply_edges",
                    "display_name_speakers_unverified",
                    "source_rights_unverified"
                ],

                "privacy_flags": sorted(privacy_flags),
                "review_status": status,
                "license_status": "verification_pending",
                "trainable": False
            }

            conversations.append(conv)

            if isolated:
                quarantine.append(conv)
            else:
                review.append(conv)

            for t in turns:
                messages.append({
                    "message_id": anon(
                        key, "path_turn",
                        f"{cid}:{t['message_order']}"
                    ),
                    "conversation_id": cid,
                    "source_group_id": group,
                    "speaker_id": t["speaker_id"],
                    "message_order": t["message_order"],
                    "text": t["text"],

                    # Flat path order is NOT a verified parent edge.
                    "parent_comment_id": None,
                    "root_comment_id": None,
                    "relation_verification": "unverified",

                    "privacy_flags": t["privacy_flags"],
                    "review_status": status,
                    "trainable": False
                })

    out.mkdir(parents=True, exist_ok=False, mode=0o700)
    os.chmod(out, 0o700)

    write_jsonl(out / "v5_paths.jsonl", conversations)
    write_jsonl(out / "v5_messages.jsonl", messages)
    write_jsonl(out / "v5_quarantine.jsonl", quarantine)

    csv_path = out / "v5_review.csv"

    with csv_path.open(
        "x", encoding="utf-8-sig", newline=""
    ) as f:
        w = csv.writer(f)

        w.writerow([
            "conversation_id",
            "turn_index",
            "speaker_id",
            "text",
            "topics",
            "trainable"
        ])

        for conv in review:
            for turn in conv["turns"]:
                w.writerow([
                    protect_csv(conv["conversation_id"]),
                    turn["message_order"],
                    protect_csv(turn["speaker_id"]),
                    protect_csv(turn["text"]),
                    protect_csv(",".join(conv["topics"])),
                    False
                ])

    os.chmod(csv_path, 0o600)

    manifest = {
        "source": "FunnySaltyFish/bilibili_comments_crawl",
        "source_type": "public_reconstructed_dialogue_examples",
        "rights": "educational/non-commercial notice; underlying rights unverified",
        "source_files": len(files),
        "paths_retained": len(conversations),
        "utterances_in_paths": len(messages),
        "paths_quarantined": len(quarantine),
        "paths_for_review": len(review),
        "invalid_or_duplicate_paths": rejected,
        "verified_parent_edges": 0,
        "trainable": False,
        "original_files_modified": False,
        "sources": source_info
    }

    with (out / "v5_manifest.json").open(
        "x", encoding="utf-8"
    ) as f:
        json.dump(
            manifest, f,
            ensure_ascii=False, indent=2
        )

    os.chmod(out / "v5_manifest.json", 0o600)

    print(
        json.dumps(
            {k: v for k, v in manifest.items() if k != "sources"},
            ensure_ascii=False,
            indent=2
        )
    )

    print("OUTPUT:", out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)

    args = parser.parse_args()

    run(args.input_dir, args.output_dir)
