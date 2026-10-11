#!/usr/bin/env python3
"""Schema probe for arbitrary JSON and JSONL datasets.

The probe reports observed keys and likely semantic roles; it never converts data.
"""
import argparse
import json
from pathlib import Path
from typing import Any, Iterable

ROLE_WORDS = {
    "dialogue": ("messages", "conversation", "conversations", "dialogue", "turns", "prompt", "response", "instruction", "output"),
    "comment": ("comment", "comments", "reply", "replies", "content", "body", "text", "评论", "回复"),
    "user": ("user", "username", "user_id", "author", "author_name", "uid", "昵称", "用户"),
    "text": ("text", "content", "body", "message", "utterance", "query", "response", "answer", "文本"),
}

def records(path: Path) -> Iterable[Any]:
    if path.suffix.lower() == ".jsonl":
        with path.open(encoding="utf-8-sig") as f:
            for line in f:
                if line.strip():
                    try: yield json.loads(line)
                    except json.JSONDecodeError: continue
    else:
        with path.open(encoding="utf-8-sig") as f:
            obj = json.load(f)
        yield from (obj if isinstance(obj, list) else [obj])

def flatten(obj: Any, prefix="") -> Iterable[tuple[str, Any]]:
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            yield key, v
            yield from flatten(v, key)
    elif isinstance(obj, list):
        for v in obj[:3]: yield from flatten(v, prefix + "[]")

def probe(path: str, sample_limit=100) -> dict:
    p = Path(path)
    sample = list(records(p))[:sample_limit]
    keys = set(); role_keys = {r: [] for r in ROLE_WORDS}
    for item in sample:
        for key, value in flatten(item):
            keys.add(key)
            low = key.lower()
            for role, words in ROLE_WORDS.items():
                if any(w.lower() in low for w in words) and key not in role_keys[role]: role_keys[role].append(key)
    return {"file": str(p), "format": p.suffix.lower().lstrip("."), "records_sampled": len(sample),
            "observed_fields": sorted(keys), "possible_fields": role_keys,
            "notes": "Roles are heuristic matches over observed keys; inspect samples before adapting."}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="JSON or JSONL file")
    ap.add_argument("-o", "--output", default="probe_report.json")
    args = ap.parse_args()
    report = probe(args.input)
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))

if __name__ == "__main__": main()
