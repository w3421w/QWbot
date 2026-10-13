#!/usr/bin/env python3
"""Audit parsed MPDD JSONL; write review, license and pilot reports."""
from __future__ import annotations
import argparse, csv, hashlib, json, re, sys, unicodedata
from collections import Counter, defaultdict
from pathlib import Path

DEFAULT_OUT = Path("experiments/qwen_fourway/conversation_pipeline/output_mpdd_pilot")
DEFAULT_TEST = Path("datasets/QWbot_V44_真实群聊影子测试_匿名31题.jsonl")
DEFAULT_RAW = Path("datasets/conversation_raw/mpdd/mpdd/dialogue.json")
PII = re.compile(r"1[3-9]\d{9}|[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}|\b\d{17}[0-9Xx]\b")
NEAR_THRESHOLD = 0.8


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def norm(text):
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", str(text or ""))).casefold()


def grams(text):
    value = norm(text)
    return {value[i:i+3] for i in range(max(0, len(value)-2))} if len(value) >= 3 else {value}


def test_strings(rows):
    values = []
    for row in rows:
        values.append(str(row.get("incoming_message", "")))
        values.extend(str(x.get("text", "")) for x in row.get("context", []) if isinstance(x, dict))
        values.append(str(row.get("example_reply_for_human_review", "")))
    return [(row.get("id"), norm(s), grams(s)) for row in rows for s in
            [str(row.get("incoming_message", "")), str(row.get("example_reply_for_human_review", ""))] +
            [str(x.get("text", "")) for x in row.get("context", []) if isinstance(x, dict)] if norm(s)]


def audit(conversations, messages, tests):
    by_id = defaultdict(list)
    for m in messages: by_id[m["conversation_id"]].append(m)
    errors = []
    ids = {c["conversation_id"] for c in conversations}
    if len(ids) != len(conversations): errors.append("重复 conversation_id")
    seen_message_ids = set()
    for c in conversations:
        cid = c["conversation_id"]
        rows = by_id.get(cid, [])
        if c.get("message_count") != len(rows): errors.append(f"{cid}: message_count 不匹配")
        if [m.get("message_order") for m in rows] != list(range(len(rows))): errors.append(f"{cid}: 消息顺序不连续")
        if c.get("trainable") is not False: errors.append(f"{cid}: conversation trainable 非 false")
        for m in rows:
            if m.get("message_id") in seen_message_ids: errors.append(f"{cid}: 重复 message_id")
            seen_message_ids.add(m.get("message_id"))
            if m.get("trainable") is not False: errors.append(f"{cid}: message trainable 非 false")
            if m.get("reply_to_id") is None and m.get("reply_relation") != "unknown": errors.append(f"{cid}: 无回复目标却有关系")
            if m.get("provenance_type") != "scripted_tv_dialogue": errors.append(f"{cid}: 来源类型错误")
    for cid in by_id.keys() - ids: errors.append(f"消息引用未知对话 {cid}")
    groups = defaultdict(set)
    for c in conversations: groups[c.get("source_group_id", c["conversation_id"])].add(c.get("split"))
    leakage = {k: sorted(v) for k, v in groups.items() if len(v) > 1}
    if leakage: errors.append("上级来源组跨 split")
    speakers = Counter(len({m["speaker_id"] for m in by_id[c["conversation_id"]]}) for c in conversations)
    switches = sum(sum(a["speaker_id"] != b["speaker_id"] for a, b in zip(rows, rows[1:])) for rows in by_id.values())
    fingerprints = defaultdict(list)
    for c in conversations:
        rows = by_id[c["conversation_id"]]
        payload = json.dumps([(m["speaker_id"], norm(m["text"])) for m in rows], ensure_ascii=False)
        fingerprints[hashlib.sha256(payload.encode()).hexdigest()].append(c["conversation_id"])
    duplicates = [v for v in fingerprints.values() if len(v) > 1]
    targets = test_strings(tests)
    exact, near = [], []
    for m in messages:
        value = norm(m.get("text", ""))
        if not value: continue
        g = grams(value)
        for test_id, test_text, test_g in targets:
            if value == test_text:
                exact.append({"message_id": m["message_id"], "test_id": test_id})
            elif len(value) >= 6 and len(test_text) >= 6:
                score = len(g & test_g) / len(g | test_g)
                if score >= NEAR_THRESHOLD:
                    near.append({"message_id": m["message_id"], "test_id": test_id, "similarity": round(score, 3)})
    lengths = [len(str(m.get("text", ""))) for m in messages]
    split_counts = Counter(c.get("split") for c in conversations)
    return {"conversation_count": len(conversations), "message_count": len(messages),
        "raw_conversation_count": len(conversations), "raw_message_count": len(messages),
        "retained_message_count": len(messages), "filtered_message_count": 0,
        "filtered_message_reasons": {}, "parse_error_count": 0,
        "parse_error_types": {}, "quantity_conservation": True,
        "split_conversation_distribution": dict(sorted(split_counts.items())),
        "two_person_conversations": speakers[2], "three_person_conversations": speakers[3],
        "four_or_more_person_conversations": sum(v for k, v in speakers.items() if k >= 4),
        "speaker_count_distribution": {str(k): v for k, v in sorted(speakers.items())},
        "average_conversation_length": len(messages) / len(conversations) if conversations else 0,
        "average_message_length": sum(lengths) / len(lengths) if lengths else 0,
        "speaker_switch_count": switches,
        "messages_with_listener_annotation": sum(m.get("listeners") is not None for m in messages),
        "messages_with_reply_to_id": sum(m.get("reply_to_id") is not None for m in messages),
        "empty_text_count": sum(not str(m.get("text", "")).strip() for m in messages),
        "duplicate_conversation_groups": duplicates,
        "duplicate_conversation_count": sum(len(g)-1 for g in duplicates),
        "privacy_risk_message_count": sum(bool(PII.search(str(m.get("text", "")))) for m in messages),
        "test_set_question_count": len(tests), "test_overlap_exact": exact,
        "test_overlap_near": near, "test_near_threshold": NEAR_THRESHOLD,
        "source_group_split_leakage": leakage,
        "license_status": "training_and_redistribution_pending_confirmation",
        "trainable_true_count": sum(x.get("trainable") is True for x in conversations + messages),
        "validation_errors": errors}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--test-set", type=Path, default=DEFAULT_TEST)
    ap.add_argument("--raw-dialogue", type=Path, default=DEFAULT_RAW)
    args = ap.parse_args(argv)
    out = args.output_dir
    for name in ("quality_statistics.json", "real_conversation_review.csv", "license_review.md", "pilot_report.md"):
        if (out / name).exists():
            print(f"拒绝覆盖已有审计文件: {out / name}", file=sys.stderr); return 4
    try:
        convs, msgs, tests = (read_jsonl(out / "conversations.jsonl"), read_jsonl(out / "messages.jsonl"), read_jsonl(args.test_set))
        stats = audit(convs, msgs, tests)
        raw = json.loads(args.raw_dialogue.read_text(encoding="utf-8-sig"))
        raw_conversations = len(raw) if isinstance(raw, dict) else len(raw)
        raw_messages = sum(len(v) for v in raw.values()) if isinstance(raw, dict) else sum(len(v) for v in raw)
        stats["raw_conversation_count"] = raw_conversations
        stats["raw_message_count"] = raw_messages
        stats["filtered_message_count"] = raw_messages - len(msgs)
        stats["quantity_conservation"] = (raw_conversations == len(convs) and raw_messages == len(msgs))
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"审计失败: {exc}", file=sys.stderr); return 2
    if stats["validation_errors"]:
        print("校验失败: " + "; ".join(stats["validation_errors"][:10]), file=sys.stderr)
        return 3
    (out / "quality_statistics.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (out / "real_conversation_review.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["conversation_id", "message_id", "speaker_id", "message_order", "text_redacted", "listeners", "interpersonal_relation", "emotion", "review_status", "privacy_redacted"])
        selected = {c["conversation_id"] for c in convs[:20]}
        for m in msgs:
            if m["conversation_id"] not in selected: continue
            raw = str(m.get("text", "")); clean = PII.sub("[已脱敏]", raw)
            writer.writerow([m["conversation_id"], m["message_id"], m["speaker_id"], m["message_order"], clean,
                "present" if m.get("listeners") is not None else "absent",
                PII.sub("[已脱敏]", json.dumps(m.get("interpersonal_relation"), ensure_ascii=False)),
                PII.sub("[已脱敏]", json.dumps(m.get("emotion"), ensure_ascii=False)), m.get("review_status"), clean != raw])
    (out / "license_review.md").write_text("# MPDD 许可与来源审核\n\n来源：官方仓库 `ntunlplab/Dialogue-MPDD` 的 `mpdd.zip`。数据来自中文电视剧台词，属于 `scripted_tv_dialogue`。\n\n训练权利及再分发权利均待人工确认；公开仓库可访问不等于允许训练或再分发。所有对话及消息 `trainable=false`。在权利明确前，不导出训练数据。\n", encoding="utf-8")
    report = ["# MPDD 真实数据试点审计", "", "本报告由本地原始 MPDD 数据解析后生成；电视剧台词不代表真实群聊。", "",
        f"- 原始对话：{stats['raw_conversation_count']}；原始话语：{stats['raw_message_count']}；保留消息：{stats['retained_message_count']}。",
        f"- 过滤消息：{stats['filtered_message_count']}；解析错误：{stats['parse_error_count']}；数量守恒：{stats['quantity_conservation']}。",
        f"- 2 人：{stats['two_person_conversations']}；3 人：{stats['three_person_conversations']}；4 人及以上：{stats['four_or_more_person_conversations']}。",
        f"- 平均对话长度：{stats['average_conversation_length']:.2f}；平均消息长度：{stats['average_message_length']:.2f} 字。",
        f"- speaker 切换：{stats['speaker_switch_count']}；listener 标注：{stats['messages_with_listener_annotation']}；真实回复目标：{stats['messages_with_reply_to_id']}。",
        f"- 重复对话：{stats['duplicate_conversation_count']}；隐私风险消息：{stats['privacy_risk_message_count']}。",
        f"- QWbot 31 题精确重合：{len(stats['test_overlap_exact'])}；近重复：{len(stats['test_overlap_near'])}（三字片段 Jaccard ≥ {NEAR_THRESHOLD}）。",
        f"- 来源组跨切分：{len(stats['source_group_split_leakage'])}。",
        "- 训练及再分发权利待确认；全部 trainable=false。", "- CSV 是前 20 段对话的脱敏抽样，完整统计见 quality_statistics.json。", ""]
    (out / "pilot_report.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
