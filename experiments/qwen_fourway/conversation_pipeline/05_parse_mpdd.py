#!/usr/bin/env python3
"""Parse MPDD dialogue.json and metadata.json from an inspected local ZIP or directory."""
from __future__ import annotations
import argparse, hashlib, json, sys, zipfile
from pathlib import Path

DEFAULT_RAW = Path("datasets/conversation_raw/mpdd")
DEFAULT_OUT = Path("experiments/qwen_fourway/conversation_pipeline/output_mpdd_pilot")
DEFAULT_MAX_SINGLE_BYTES = 128 * 1024 * 1024
SOURCE = "Dialogue-MPDD"
FIELDS = ("utterances", "messages", "dialogue", "conversation", "turns")
ID_FIELDS = ("conversation_id", "dialogue_id", "dialog_id", "id")
TEXT_FIELDS = ("text", "utterance", "content", "sentence")
SPEAKER_FIELDS = ("speaker_id", "speaker", "role")
LISTENER_FIELDS = ("listeners", "listener", "listener_id")
PARENT_FIELDS = ("drama_id", "tv_show_id", "show_id", "series_id", "drama", "tv_show", "series", "episode_id", "episode", "scene_id", "scene")


def first(obj, keys, default=None):
    return next((obj[k] for k in keys if k in obj), default)


def load_inputs(input_dir: Path):
    zip_path = input_dir / "mpdd.zip"
    if zip_path.exists():
        import importlib.util
        spec = importlib.util.spec_from_file_location("mpdd_fetch", Path(__file__).with_name("04_fetch_mpdd.py"))
        fetch = importlib.util.module_from_spec(spec); spec.loader.exec_module(fetch)
        fetch.validate_zip_bytes(zip_path.read_bytes())
        with zipfile.ZipFile(zip_path) as archive:
            names = archive.namelist()
            result = {}
            for target in ("dialogue.json", "metadata.json"):
                matches = [name for name in names if Path(name).name.lower() == target]
                if len(matches) != 1: raise ValueError(f"ZIP 中应恰有一个 {target}，实际 {len(matches)} 个")
                result[target] = json.loads(archive.read(matches[0]).decode("utf-8-sig"))
            return result, {"dialogue.json": f"{zip_path}!/{[n for n in names if Path(n).name.lower() == 'dialogue.json'][0]}",
                            "metadata.json": f"{zip_path}!/{[n for n in names if Path(n).name.lower() == 'metadata.json'][0]}"}
    result = {}
    paths = {}
    for target in ("dialogue.json", "metadata.json"):
        matches = list(input_dir.rglob(target))
        if len(matches) != 1: raise ValueError(f"目录中应恰有一个 {target}，实际 {len(matches)} 个")
        result[target] = json.loads(matches[0].read_text(encoding="utf-8-sig"))
        paths[target] = str(matches[0])
    return result, paths


def safe_extract_zip(zip_path: Path, output_dir: Path, max_single_bytes: int = DEFAULT_MAX_SINGLE_BYTES):
    """Extract a ZIP without traversal, symlink, overwrite, or size surprises.

    Existing regular files are retained only when their bytes exactly match the
    archive member; all other collisions are rejected.
    """
    import os
    import stat
    import tempfile
    import importlib.util
    spec = importlib.util.spec_from_file_location("mpdd_fetch", Path(__file__).with_name("04_fetch_mpdd.py"))
    fetch = importlib.util.module_from_spec(spec); spec.loader.exec_module(fetch)
    blob = zip_path.read_bytes()
    fetch.validate_zip_bytes(blob, max_uncompressed_bytes=512 * 1024 * 1024,
                             max_single_bytes=max_single_bytes)
    output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        for info in archive.infolist():
            name = info.filename.replace("\\", "/")
            if info.is_dir():
                continue
            mode = (info.external_attr >> 16) & 0o170000
            if mode == stat.S_IFLNK:
                raise ValueError(f"ZIP 符号链接被拒绝: {info.filename}")
            if not fetch.zip_member_is_safe(name):
                raise ValueError(f"ZIP 路径不安全: {info.filename}")
            target = (output_dir / name).resolve()
            root = output_dir.resolve()
            if target != root and root not in target.parents:
                raise ValueError(f"ZIP 路径越界: {info.filename}")
            data = archive.read(info)
            if len(data) != info.file_size or len(data) > max_single_bytes:
                raise ValueError(f"ZIP 文件大小校验失败: {info.filename}")
            if target.exists():
                if not target.is_file() or target.read_bytes() != data:
                    raise FileExistsError(f"拒绝覆盖不一致文件: {target}")
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=str(target.parent))
            try:
                with os.fdopen(fd, "wb") as handle:
                    handle.write(data)
                os.replace(temporary, target)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)


def dialogue_records(obj):
    if isinstance(obj, dict):
        for key in ("dialogues", "conversations", "data", "dialogs"):
            if isinstance(obj.get(key), (list, dict)):
                return dialogue_records(obj[key])
        if any(isinstance(obj.get(k), list) for k in FIELDS):
            return [(None, obj)]
        if obj and all(isinstance(v, (list, dict)) for v in obj.values()):
            return list(obj.items())
    if isinstance(obj, list):
        return [(None, item) for item in obj]
    raise ValueError("无法识别 dialogue.json 的对话容器；请核对官方格式")


def turns_for(record):
    if isinstance(record, list): return record
    if isinstance(record, dict):
        turns = first(record, FIELDS)
        if isinstance(turns, list): return turns
        # A dict of order -> utterance is only accepted when every key is numeric.
        if isinstance(turns, dict) and all(str(k).isdigit() for k in turns):
            return [turns[k] for k in sorted(turns, key=lambda k: int(k))]
    raise ValueError("对话缺少可识别的话语列表")


def metadata_for(metadata, raw_id, index):
    if isinstance(metadata, list):
        return metadata[index] if index < len(metadata) and isinstance(metadata[index], dict) else {}
    if isinstance(metadata, dict):
        for key in ("metadata", "dialogues", "conversations", "data"):
            if isinstance(metadata.get(key), (list, dict)):
                return metadata_for(metadata[key], raw_id, index)
        value = metadata.get(str(raw_id)) if raw_id is not None else None
        if isinstance(value, dict): return value
        if any(k in metadata for k in PARENT_FIELDS): return metadata
    return {}


def parent_group(record, meta, cid):
    # Use the broadest actually present source identifier for leakage control.
    for keys in (("drama_id", "tv_show_id", "show_id", "series_id"),
                 ("drama", "tv_show", "series"), ("episode_id", "episode"),
                 ("scene_id", "scene")):
        value = first(record, keys) if isinstance(record, dict) else None
        if value is None: value = first(meta, keys)
        if value is not None and str(value).strip(): return f"{keys[0]}:{value}"
    return cid


def parse(dialogue, metadata, source_file="dialogue.json"):
    result = []
    records = dialogue_records(dialogue)
    used_ids = set()
    for index, (mapping_key, record) in enumerate(records):
        turns = turns_for(record)
        if not isinstance(turns, list): raise ValueError(f"对话 {index} 的话语不是列表")
        raw_id = first(record, ID_FIELDS, mapping_key) if isinstance(record, dict) else mapping_key
        raw_id = str(raw_id) if raw_id is not None else str(index)
        cid = "mpdd:" + raw_id
        if cid in used_ids: raise ValueError(f"重复的原始对话 ID: {raw_id}")
        used_ids.add(cid)
        meta = metadata_for(metadata, raw_id, index)
        if not isinstance(meta, dict): meta = {}
        speakers, messages = {}, []
        for raw_order, item in enumerate(turns):
            if not isinstance(item, dict): raise ValueError(f"对话 {raw_id} 第 {raw_order} 条话语不是对象")
            text = first(item, TEXT_FIELDS)
            speaker = first(item, SPEAKER_FIELDS)
            if text is None or speaker is None:
                raise ValueError(f"对话 {raw_id} 第 {raw_order} 条缺少 text 或 speaker")
            # Preserve empty dialogue lines and the exact order, with no invented clock time.
            key = json.dumps(speaker, ensure_ascii=False, sort_keys=True)
            speakers.setdefault(key, f"speaker_{len(speakers)+1:02d}")
            listener = first(item, LISTENER_FIELDS)
            relation = first(item, ("interpersonal_relation", "relationship", "relation"))
            if relation is None and isinstance(listener, list):
                rels = [x.get("relation") for x in listener if isinstance(x, dict) and x.get("relation") is not None]
                if rels:
                    relation = rels[0] if len(rels) == 1 else rels
            elif relation is None and isinstance(listener, dict):
                relation = listener.get("relation")
            message_id = f"{cid}:m{raw_order:04d}"
            messages.append({"conversation_id": cid, "message_id": message_id,
                "speaker_id": speakers[key], "message_order": raw_order, "text": str(text),
                "listeners": listener, "interpersonal_relation": relation,
                "emotion": first(item, ("emotion", "emotions")),
                "reply_to_id": None, "reply_relation": "unknown", "source": SOURCE,
                "source_file": source_file, "provenance_type": "scripted_tv_dialogue",
                "is_synthetic": False, "license_status": "training_and_redistribution_pending_confirmation",
                "review_status": "pending", "trainable": False})
        group = parent_group(record, meta, cid)
        split = "validation" if int(hashlib.sha256(group.encode("utf-8")).hexdigest()[:8], 16) % 10 == 0 else "train"
        for m in messages: m["split"] = split
        result.append({"conversation_id": cid, "message_count": len(messages),
            "speaker_count": len(speakers), "speaker_ids": list(speakers.values()),
            "source_group_id": group, "split": split, "source": SOURCE,
            "source_file": source_file, "provenance_type": "scripted_tv_dialogue",
            "is_synthetic": False, "license_status": "training_and_redistribution_pending_confirmation",
            "review_status": "pending", "trainable": False, "messages": messages})
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-dir", type=Path, default=DEFAULT_RAW)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args(argv)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        print(f"拒绝覆盖非空输出目录: {args.output_dir}", file=sys.stderr); return 4
    try:
        zip_path = args.input_dir / "mpdd.zip"
        if zip_path.exists():
            safe_extract_zip(zip_path, args.input_dir)
        inputs, paths = load_inputs(args.input_dir)
        conversations = parse(inputs["dialogue.json"], inputs["metadata.json"], paths["dialogue.json"])
    except (OSError, ValueError, json.JSONDecodeError, zipfile.BadZipFile) as exc:
        print(f"MPDD 解析失败: {exc}", file=sys.stderr); return 2
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "conversations.jsonl").write_text("".join(json.dumps({k: v for k, v in c.items() if k != "messages"}, ensure_ascii=False) + "\n" for c in conversations), encoding="utf-8")
    (args.output_dir / "messages.jsonl").write_text("".join(json.dumps(m, ensure_ascii=False) + "\n" for c in conversations for m in c["messages"]), encoding="utf-8")
    print(f"解析完成: {len(conversations)} 段对话，{sum(c['message_count'] for c in conversations)} 条话语；全部 trainable=false")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
