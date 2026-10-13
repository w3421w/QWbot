#!/usr/bin/env python3
"""Parse SmoothConv JSON annotations into anonymized, reviewable conversation JSONL."""
from __future__ import annotations
import argparse, hashlib, json, re
from pathlib import Path

LICENSE = "CC BY-NC 4.0"
PII = re.compile(r"(?:1[3-9]\d{9}|[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}|(?:身份证|证件)号?\s*[:：]?\s*[0-9Xx]{6,})")
GARBAGE = re.compile(r"^(?:嗯+|啊+|呃+|<unk>|\[unk\]|noise|silence|\.{3,})$", re.I)

def load_records(path):
    obj = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(obj, dict) and isinstance(obj.get("instances"), list): return [obj]
    if isinstance(obj, list): return [x for x in obj if isinstance(x, dict)]
    if isinstance(obj, dict):
        for key in ("data", "annotations", "conversations"):
            if isinstance(obj.get(key), list): return [x for x in obj[key] if isinstance(x, dict)]
    return []

def _source_key(path: Path, source_root: Path | None = None) -> str:
    """Return a stable path key so same-named files in different dirs differ."""
    path = path.resolve()
    if source_root is not None:
        try:
            return path.relative_to(source_root.resolve()).as_posix()
        except ValueError:
            pass
    return path.as_posix()

def parse_file(path: Path, source_root: Path | None = None):
    out, errors = [], []
    try: records = load_records(path)
    except Exception as e:
        return [], [{"source_file": str(path), "conversation_index": None,
                     "instance_index": None, "raw_field_path": None,
                     "error_type": "json_load_error", "message": str(e),
                     "lost_message": False, "affects_speaker_id": False,
                     "affects_time_order": False}]
    for idx, rec in enumerate(records):
        instances = rec.get("instances")
        if not isinstance(instances, list):
            errors.append({"source_file": str(path), "conversation_index": idx,
                           "instance_index": None, "raw_field_path": "instances",
                           "error_type": "missing_instances_list", "message": "缺少 instances 列表",
                           "lost_message": False, "affects_speaker_id": False,
                           "affects_time_order": False})
            continue
        source_key = _source_key(path, source_root)
        source_hash = hashlib.sha256(source_key.encode("utf-8")).hexdigest()[:12]
        cid = f"smoothconv:{path.stem}:{source_hash}:{idx}"
        speakers, rows = {}, []
        for raw_i, item in enumerate(instances):
            if not isinstance(item, dict):
                errors.append({"source_file": str(path), "conversation_index": idx,
                               "instance_index": raw_i, "raw_field_path": None,
                               "error_type": "instance_not_object", "message": "instance 不是对象",
                               "lost_message": True, "affects_speaker_id": True,
                               "affects_time_order": True})
                continue
            attrs = item.get("attributes")
            if not isinstance(attrs, dict): attrs = {}
            speaker = attrs.get("speaker", item.get("channelIndex"))
            text = item.get("text")
            start, end = item.get("start"), item.get("end")
            missing = [name for name, value in (("speaker", speaker), ("text", text), ("start", start), ("end", end)) if value is None]
            if missing:
                errors.append({"source_file": str(path), "conversation_index": idx,
                               "instance_index": raw_i,
                               "raw_field_path": ["attributes.speaker" if x == "speaker" else x for x in missing],
                               "error_type": "missing_required_field", "missing_fields": missing,
                               "message": "字段不完整", "lost_message": True,
                               "affects_speaker_id": speaker is None,
                               "affects_time_order": start is None or end is None})
                continue
            text = str(text).strip()
            if not text or GARBAGE.fullmatch(text): continue
            key = str(speaker)
            speakers.setdefault(key, f"speaker_{len(speakers)+1:02d}")
            flags = []
            if PII.search(text): flags.append("privacy_risk")
            if len(text) > 2000: flags.append("very_long")
            rows.append((raw_i, speaker, float(start), float(end), text, flags))
        rows.sort(key=lambda x: (x[2], x[3], x[0]))
        if not rows: continue
        # Stable hash split keeps a conversation wholly in one partition.
        split = "validation" if int(hashlib.sha256(cid.encode()).hexdigest()[:2], 16) < 26 else "train"
        msgs = []
        for order, (_, sp, start, end, text, flags) in enumerate(rows):
            msgs.append({"conversation_id": cid, "message_id": f"{cid}:m{order:04d}", "speaker_id": speakers[str(sp)], "message_order": order, "start_time": start, "end_time": end, "text": text, "reply_to_id": None, "reply_relation": "unknown", "topic": rec.get("topic"), "source": "SmoothConv", "source_file": str(path), "source_license": LICENSE, "license_status": "non_commercial_research_only", "provenance_type": "speech_transcript", "is_synthetic": False, "quality_flags": flags, "review_status": "pending", "trainable": False, "split": split})
        out.append({"conversation_id": cid, "message_count": len(msgs), "speaker_count": len(speakers), "speaker_ids": sorted({m["speaker_id"] for m in msgs}), "split": split, "source_file": str(path), "source": "SmoothConv", "source_license": LICENSE, "license_status": "non_commercial_research_only", "provenance_type": "speech_transcript", "has_real_reply_targets": False, "trainable": False, "messages": msgs})
    return out, errors

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-dir", type=Path, default=Path("datasets/conversation_raw/smoothconv_json"))
    ap.add_argument("--output-dir", type=Path, default=Path("experiments/qwen_fourway/conversation_pipeline/output_smoothconv_pilot"))
    args = ap.parse_args(argv); args.output_dir.mkdir(parents=True, exist_ok=True)
    conversations, errors = [], []
    for p in sorted(args.input_dir.rglob("*.json")):
        if p.name in {"download_manifest.json"} or p.name.endswith(".source.json"): continue
        c, e = parse_file(p, args.input_dir); conversations.extend(c); errors.extend(e)
    with (args.output_dir / "conversations.jsonl").open("w", encoding="utf-8") as f:
        for c in conversations: f.write(json.dumps({k:v for k,v in c.items() if k != "messages"}, ensure_ascii=False) + "\n")
    with (args.output_dir / "messages.jsonl").open("w", encoding="utf-8") as f:
        for c in conversations:
            for m in c["messages"]: f.write(json.dumps(m, ensure_ascii=False) + "\n")
    (args.output_dir / "parse_errors.json").write_text(json.dumps(errors, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"解析完成: conversations={len(conversations)}, messages={sum(c['message_count'] for c in conversations)}, errors={len(errors)}")
    return 0
if __name__ == "__main__": raise SystemExit(main())
