#!/usr/bin/env python3
"""QWbot V5: bounded public Bilibili conversation pilot."""
import argparse
import csv
import hashlib
import hmac
import json
import os
import secrets
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from datetime import datetime, timezone

P = Path(__file__).resolve().parent
sys.path.insert(0, str(P))

from common.core import privacy

SEEDS = [
    ("anti_ba_discussion", "BV1XH4y1B7Y3"),
    ("lol", "BV1WY4gzRE25"),
    ("cs2", "BV1TnmiBaEGD"),
    ("valorant", "BV1p2xMz5EQm"),
    ("hearthstone", "BV1MebPzmEwG"),
    ("cn_vup", "BV1vWTLzBEx5"),
]

FUTURE = [
    "dota2", "nba", "football",
    "jp_vtuber", "zh_meme", "everyday_chat"
]

BLOCK_CODES = {-403, -412, -352, -101, -102, -509}


class ApiBlocked(Exception):
    pass


def request_json(path, **params):
    time.sleep(1.3)

    url = (
        "https://api.bilibili.com" + path + "?"
        + urllib.parse.urlencode(params)
    )

    req = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "QWbot-v5-research-pilot/1.0"
        }
    )

    try:
        with urllib.request.urlopen(req, timeout=12) as response:
            if response.status != 200:
                raise ApiBlocked(
                    f"HTTP {response.status}"
                )
            raw = response.read(1_000_001)

    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403, 412, 429):
            raise ApiBlocked(
                f"HTTP {exc.code}"
            ) from exc
        raise RuntimeError(
            f"HTTP {exc.code}"
        ) from exc

    except (urllib.error.URLError, TimeoutError) as exc:
        raise RuntimeError(
            f"Network unavailable: {exc}"
        ) from exc

    if len(raw) > 1_000_000:
        raise RuntimeError("Response exceeds size limit")

    obj = json.loads(raw)

    if not isinstance(obj, dict):
        raise RuntimeError("Invalid API response")

    code = obj.get("code")

    if code in BLOCK_CODES:
        raise ApiBlocked(
            f"Bilibili API code={code}"
        )

    if code != 0:
        raise RuntimeError(
            f"Bilibili API code={code}"
        )

    return obj.get("data") or {}


def load_salt():
    """Keep the pseudonymization key outside the dataset."""
    path = Path.home() / ".config/qwbot/v5_hmac.key"

    path.parent.mkdir(
        parents=True, exist_ok=True, mode=0o700
    )

    if not path.exists():
        try:
            fd = os.open(
                path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600
            )
            with os.fdopen(fd, "wb") as f:
                f.write(secrets.token_bytes(32))
        except FileExistsError:
            pass

    if path.stat().st_mode & 0o077:
        raise RuntimeError("Unsafe key permissions")

    salt = path.read_bytes()

    if len(salt) != 32:
        raise RuntimeError("Invalid key")

    return salt


def anon(salt, kind, value):
    message = f"{kind}:{value}".encode()
    return hmac.new(
        salt, message, hashlib.sha256
    ).hexdigest()[:24]


def comment_id(salt, aid, rpid):
    return "bili:" + anon(
        salt, "comment", f"{aid}:{rpid}"
    )


def normalize(comment, salt, aid, bvid, topic, title):
    if not isinstance(comment, dict):
        return None

    rid = comment.get("rpid_str") or comment.get("rpid")
    content = comment.get("content") or {}

    if not rid or not isinstance(content, dict):
        return None

    text = content.get("message")

    if not isinstance(text, str) or not text.strip():
        return None

    safe_text, flags = privacy(text.strip())

    parent = (
        comment.get("parent_str")
        or comment.get("parent")
        or 0
    )
    root = (
        comment.get("root_str")
        or comment.get("root")
        or 0
    )

    parent_id = (
        comment_id(salt, aid, parent)
        if str(parent) not in ("0", "", "None")
        else None
    )

    root_id = (
        comment_id(salt, aid, root)
        if str(root) not in ("0", "", "None")
        else None
    )

    member = comment.get("member") or {}
    mid = comment.get("mid") or member.get("mid")

    quarantined = bool(
        {"ambiguous_mention", "private_identity"}
        & set(flags)
    )

    return {
        "platform": "bilibili",
        "domain": ["chinese_community"],
        "topics": [topic],
        "video_id": bvid,
        "video_title": privacy(title)[0],
        "thread_id": f"bili-video:{bvid}",
        "message_id": comment_id(salt, aid, rid),
        "parent_comment_id": parent_id,
        "root_comment_id": root_id,
        "speaker_id": (
            "bili-user:" + anon(salt, "speaker", mid)
            if mid else None
        ),
        "text": safe_text,
        "created_at": comment.get("ctime"),
        "source_url": (
            "https://www.bilibili.com/video/" + bvid
        ),
        "source_record_id": anon(
            salt, "comment", f"{aid}:{rid}"
        ),
        "content_type": "comment",
        "privacy_flags": flags,
        "quality_flags": [
            "small_public_pilot",
            "incomplete_comment_tree",
            "rights_unverified"
        ],
        "review_status": (
            "privacy_quarantine"
            if quarantined else "unreviewed"
        ),
        "license_status": "verification_pending",
        "source_license": "unknown",
        "provenance_type": "bilibili_public_api_pilot",
        "trainable": False
    }


def collect_video(topic, bvid, salt, nested_roots):
    view = request_json(
        "/x/web-interface/view",
        bvid=bvid
    )

    aid = view.get("aid")

    if not aid:
        raise RuntimeError("Missing video aid")

    title = str(view.get("title") or "")

    first = request_json(
        "/x/v2/reply/main",
        type=1,
        oid=aid,
        mode=3,
        next=0,
        ps=20
    )

    roots = first.get("replies") or []

    if not isinstance(roots, list):
        raise RuntimeError("Invalid replies list")

    indexed = {}

    def add(item):
        row = normalize(
            item, salt, aid, bvid, topic, title
        )
        if row:
            indexed.setdefault(
                row["message_id"], row
            )

    for root in roots:
        add(root)

        if isinstance(root, dict):
            for child in root.get("replies") or []:
                add(child)

    for root in roots[:nested_roots]:
        if not isinstance(root, dict):
            continue

        if not root.get("rcount"):
            continue

        rid = (
            root.get("rpid_str")
            or root.get("rpid")
        )

        if not rid:
            continue

        data = request_json(
            "/x/v2/reply/reply",
            type=1,
            oid=aid,
            root=rid,
            ps=20,
            pn=1
        )

        for child in data.get("replies") or []:
            add(child)

    records = list(indexed.values())
    known = set(indexed)

    for row in records:
        parent = row["parent_comment_id"]

        if parent and parent not in known:
            row["quality_flags"].append(
                "parent_out_of_scope"
            )

    stats = {
        "topic": topic,
        "bvid": bvid,
        "title": privacy(title)[0],
        "records": len(records),
        "real_parent_edges": sum(
            bool(r["parent_comment_id"])
            for r in records
        ),
        "in_scope_parent_edges": sum(
            r["parent_comment_id"] in known
            for r in records
        ),
        "quarantined": sum(
            r["review_status"] == "privacy_quarantine"
            for r in records
        )
    }

    return records, stats


def self_test():
    salt = b"1" * 32

    root = {
        "rpid": 10,
        "parent": 0,
        "root": 0,
        "mid": 101,
        "content": {"message": "抗吧热议"}
    }

    reply = {
        "rpid": 11,
        "parent": 10,
        "root": 10,
        "mid": 102,
        "content": {
            "message": "回复 @小明 :这波不亏"
        }
    }

    a = normalize(
        root, salt, 100, "BV_TEST",
        "lol", "LOL分析"
    )
    b = normalize(
        reply, salt, 100, "BV_TEST",
        "lol", "LOL分析"
    )

    assert a["message_id"] == b["parent_comment_id"]
    assert a["message_id"] == b["root_comment_id"]
    assert a["speaker_id"] != b["speaker_id"]
    assert "小明" not in json.dumps(
        b, ensure_ascii=False
    )
    assert b["text"] == "回复 @用户 :这波不亏"
    assert a["trainable"] is False
    assert b["trainable"] is False

    ambiguous = normalize(
        {
            "rpid": 12,
            "parent": 0,
            "content": {
                "message": "今天@张三和@李四都来了"
            }
        },
        salt, 100, "BV_TEST",
        "lol", ""
    )

    assert (
        ambiguous["review_status"]
        == "privacy_quarantine"
    )

    print("V5_SELF_TEST_OK")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--self-test", action="store_true"
    )
    parser.add_argument(
        "--plan", action="store_true"
    )
    parser.add_argument(
        "--start", type=int, default=0
    )
    parser.add_argument(
        "--max-videos", type=int, default=3
    )
    parser.add_argument(
        "--nested-roots", type=int, default=3
    )
    parser.add_argument(
        "--output-dir", type=Path
    )

    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    if args.plan:
        for topic, bvid in SEEDS:
            print(topic, bvid)
        print("Future:", ", ".join(FUTURE))
        return

    if not 0 <= args.start < len(SEEDS):
        parser.error("--start must be 0..5")

    if not 1 <= args.max_videos <= 6:
        parser.error("--max-videos must be 1..6")

    if not 0 <= args.nested_roots <= 5:
        parser.error("--nested-roots must be 0..5")

    if args.output_dir is None:
        parser.error("--output-dir is required")

    out = args.output_dir.resolve()

    if out.exists():
        parser.error(
            f"Refusing to overwrite: {out}"
        )

    out.mkdir(
        parents=True,
        exist_ok=False,
        mode=0o700
    )
    os.chmod(out, 0o700)

    salt = load_salt()

    all_rows = []
    stats = []
    failures = []

    batch = SEEDS[
        args.start:args.start + args.max_videos
    ]

    for topic, bvid in batch:
        try:
            rows, info = collect_video(
                topic, bvid, salt,
                args.nested_roots
            )

            all_rows.extend(rows)
            stats.append(info)

            print(
                f"{topic}: {len(rows)} records, "
                f"{info['in_scope_parent_edges']} reply edges"
            )

        except ApiBlocked as exc:
            failures.append({
                "topic": topic,
                "error": str(exc)
            })
            print(
                "API_BLOCKED: stopping without bypass",
                exc,
                file=sys.stderr
            )
            break

        except Exception as exc:
            failures.append({
                "topic": topic,
                "error": str(exc)
            })
            print(
                "SOURCE_SKIPPED:",
                topic, exc,
                file=sys.stderr
            )

    with (
        out / "v5_candidates.jsonl"
    ).open("x", encoding="utf-8") as f:
        for row in all_rows:
            f.write(
                json.dumps(row, ensure_ascii=False)
                + "\n"
            )

    with (
        out / "v5_quarantine.jsonl"
    ).open("x", encoding="utf-8") as f:
        for row in all_rows:
            if row["review_status"] == "privacy_quarantine":
                f.write(
                    json.dumps(row, ensure_ascii=False)
                    + "\n"
                )

    fields = [
        "topics",
        "video_id",
        "message_id",
        "parent_comment_id",
        "speaker_id",
        "text",
        "privacy_flags",
        "review_status",
        "license_status",
        "trainable"
    ]

    with (
        out / "v5_review.csv"
    ).open(
        "x", encoding="utf-8-sig", newline=""
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=fields
        )
        writer.writeheader()

        for row in all_rows:
            if row["review_status"] == "privacy_quarantine":
                continue

            item = {
                k: row.get(k) for k in fields
            }

            for k in ("topics", "privacy_flags"):
                item[k] = json.dumps(
                    item[k], ensure_ascii=False
                )

            for k, v in item.items():
                if (
                    isinstance(v, str)
                    and v.lstrip()[:1] in ("=", "+", "-", "@")
                ):
                    item[k] = "'" + v

            writer.writerow(item)

    report = {
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "seed_start": args.start,
        "sources": stats,
        "failures": failures,
        "total_records": len(all_rows),
        "verified_replies": sum(
            s["in_scope_parent_edges"]
            for s in stats
        ),
        "quarantined": sum(
            r["review_status"] == "privacy_quarantine"
            for r in all_rows
        ),
        "next_topics": FUTURE,
        "trainable": False,
        "rights_status": "verification_pending"
    }

    with (
        out / "v5_manifest.json"
    ).open("x", encoding="utf-8") as f:
        json.dump(
            report, f,
            ensure_ascii=False, indent=2
        )

    for path in out.iterdir():
        os.chmod(path, 0o600)

    print(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2
        )
    )
    print("OUTPUT:", out)


if __name__ == "__main__":
    main()
