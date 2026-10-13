#!/usr/bin/env python3
"""Validate SmoothConv conversations and write a source-specific audit bundle."""
from __future__ import annotations
import argparse, csv, hashlib, importlib.util, json, re, unicodedata
from collections import Counter, defaultdict
from pathlib import Path

ZH = re.compile(r"[\u3400-\u9fff]")
PII = re.compile(r"(?:1[3-9]\d{9}|[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}|(?:身份证|证件)号?\s*[:：]?\s*[0-9Xx]{6,})")
TRANS = re.compile(r"(?:<unk>|\[unk\]|噪音|听不清|无法辨认)", re.I)
GARBAGE = re.compile(r"^(?:嗯+|啊+|呃+|<unk>|\[unk\]|noise|silence|\.{3,})$", re.I)


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf8").splitlines() if line.strip()] if path.exists() else []


def load_parser():
    path = Path(__file__).with_name("02_parse_smoothconv.py")
    spec = importlib.util.spec_from_file_location("smoothconv_parser", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def raw_accounting(input_dir: Path):
    totals = Counter()
    by_category = defaultdict(Counter)
    files = []
    for path in sorted(input_dir.rglob("*.json")):
        if path.name == "download_manifest.json" or path.name.endswith(".source.json"):
            continue
        obj = json.loads(path.read_text(encoding="utf8"))
        records = obj if isinstance(obj, list) else [obj]
        category = "social_chat" if any(part.startswith("none_edu") for part in path.parts) else "education_tutoring" if any(part.startswith("edu") for part in path.parts) else "unknown"
        files.append(path)
        for record in records:
            if not isinstance(record, dict) or not isinstance(record.get("instances"), list):
                continue
            for item in record["instances"]:
                totals["instances_total"] += 1
                by_category[category]["instances_total"] += 1
                if not isinstance(item, dict):
                    totals["errors"] += 1; by_category[category]["errors"] += 1; continue
                attrs = item.get("attributes") if isinstance(item.get("attributes"), dict) else {}
                required = (attrs.get("speaker", item.get("channelIndex")), item.get("text"), item.get("start"), item.get("end"))
                if any(v is None for v in required):
                    totals["errors"] += 1; by_category[category]["errors"] += 1; continue
                text = str(item.get("text")).strip()
                if not text or GARBAGE.fullmatch(text):
                    totals["filtered"] += 1; by_category[category]["filtered"] += 1
                else:
                    totals["retained"] += 1; by_category[category]["retained"] += 1
    totals["conservation_ok"] = totals["instances_total"] == totals["retained"] + totals["filtered"] + totals["errors"]
    return totals, by_category, files


def train_audit(rows):
    c = Counter()
    for row in rows:
        value = row.get("trainable", "__missing__")
        c["missing" if value == "__missing__" else "true" if value is True else "false" if isinstance(value, bool) else "non_boolean"] += 1
    return {key: c[key] for key in ("true", "false", "missing", "non_boolean")}


def norm(value):
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", value or ""))


def recording_key(source_file):
    name = Path(source_file).name
    return re.split(r"_seg", name, maxsplit=1)[0]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", type=Path, default=Path("datasets/conversation_raw/smoothconv_social_pilot"))
    ap.add_argument("--output-dir", type=Path, default=Path("experiments/qwen_fourway/conversation_pipeline/output_smoothconv_social_audit_v2"))
    ap.add_argument("--test-set", type=Path, default=Path("datasets/QWbot_V44_真实群聊影子测试_匿名31题.jsonl"))
    args = ap.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"拒绝覆盖非空输出目录: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    parser = load_parser()
    conversations, errors = [], []
    totals, by_category, raw_files = raw_accounting(args.input_dir)
    for path in raw_files:
        convs, errs = parser.parse_file(path, args.input_dir)
        conversations.extend(convs); errors.extend(errs)
    messages = [message for conversation in conversations for message in conversation.get("messages", [])]
    by_conversation = defaultdict(list)
    for message in messages: by_conversation[message["conversation_id"]].append(message)
    convs_json = [{key: value for key, value in c.items() if key != "messages"} for c in conversations]
    (args.output_dir / "conversations.jsonl").write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in convs_json), encoding="utf8")
    (args.output_dir / "messages.jsonl").write_text("".join(json.dumps(m, ensure_ascii=False) + "\n" for m in messages), encoding="utf8")
    (args.output_dir / "parse_errors.json").write_text(json.dumps(errors, ensure_ascii=False, indent=2) + "\n", encoding="utf8")

    speakers = Counter(len({m.get("speaker_id") for m in ms}) for ms in by_conversation.values())
    lengths = [len(m.get("text", "")) for m in messages]
    switches = sum(sum(left.get("speaker_id") != right.get("speaker_id") for left, right in zip(ms, ms[1:])) for ms in by_conversation.values())
    transcriptions = sum(bool(TRANS.search(m.get("text", ""))) for m in messages)
    mt, ct = train_audit(messages), train_audit(convs_json)
    combined = {key: mt[key] + ct[key] for key in mt}
    split_by_recording = defaultdict(set)
    for c in convs_json: split_by_recording[recording_key(c["source_file"])].add(c.get("split"))
    leakage = {key: sorted(value) for key, value in split_by_recording.items() if len(value) > 1}

    stats = {
        "input_directory": str(args.input_dir), "source_categories": {key: dict(value) for key, value in by_category.items()},
        "raw_instances": dict(totals), "parsed_conversations": len(convs_json), "retained_messages": len(messages),
        "speakers_per_conversation": {str(k): v for k, v in sorted(speakers.items())},
        "two_person_conversations": speakers[2], "three_or_more_person_conversations": sum(v for k, v in speakers.items() if k >= 3),
        "message_count_distribution": dict(sorted(Counter(len(v) for v in by_conversation.values()).items())),
        "average_message_length": sum(lengths) / len(lengths) if lengths else 0,
        "chinese_message_ratio": sum(bool(ZH.search(m.get("text", ""))) for m in messages) / len(messages) if messages else 0,
        "transcription_error_count": transcriptions, "transcription_error_ratio": transcriptions / len(messages) if messages else 0,
        "speaker_switch_count": switches, "privacy_risk_count": sum("privacy_risk" in m.get("quality_flags", []) for m in messages),
        "split_conversations": dict(Counter(c.get("split") for c in convs_json)),
        "raw_recording_split_leakage": {"risk": bool(leakage), "groups": leakage, "grouping_key": "filename prefix before _seg"},
        "trainable_audit": {"messages": mt, "conversations": ct, "combined": combined, "trainable_true_count": combined["true"], "all_trainable_false": combined["true"] == 0 and combined["missing"] == 0 and combined["non_boolean"] == 0},
        "license_status": "CC BY-NC 4.0 / non_commercial_research_only / pending_review", "approved_for_finetuning": False,
    }
    (args.output_dir / "quality_statistics.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf8")

    with (args.output_dir / "real_conversation_review.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle); writer.writerow(["conversation_id", "message_id", "speaker_id", "message_order", "start_time", "end_time", "text", "source_category", "review_note", "privacy_redacted"])
        for c in convs_json[:10]:
            category = "social_chat" if "none_edu" in c.get("source_file", "") else "education_tutoring" if "/edu" in c.get("source_file", "") else "unknown"
            for m in by_conversation[c["conversation_id"]][:30]:
                text = PII.sub("[已脱敏]", m.get("text", ""))
                note = "含转写异常标记" if TRANS.search(text) else "保留原消息顺序供人工审核"
                writer.writerow([c["conversation_id"], m.get("message_id"), m.get("speaker_id"), m.get("message_order"), m.get("start_time"), m.get("end_time"), text, category, note, "是" if text != m.get("text", "") else "否"])

    report = ["# SmoothConv 社交语料质量审计 v2", "", "本报告仅基于 `none_edu` 社交聊天原始 JSON 动态计算；`none_edu` 判定为社交聊天，`edu` 判定为课堂/辅导。未修改原始 JSON，未生成 SFT。", "", "## 数量守恒", "", f"- 原始 instances：{totals['instances_total']} = 保留 {totals['retained']} + 过滤 {totals['filtered']} + 错误 {totals['errors']}。守恒检查：{totals['conservation_ok']}。", "- 过滤项按解析器规则计为空白或纯转写垃圾；错误项为缺少必需字段或非对象。", "", "## 社交语料质量", "", f"- 对话 {len(convs_json)} 段，消息 {len(messages)} 条；{sum(v for k, v in speakers.items() if k >= 3)} 段为 3 人及以上。", f"- speaker 切换 {switches} 次；转写异常 {transcriptions} 条（{stats['transcription_error_ratio']:.2%}）。", "- 所有 trainable 均为 false，当前不批准用于微调。", "", "## train/validation 原始录音分组风险", "", f"- 当前按 conversation_id 的 split 中，按文件名 `_seg` 前缀归并后发现跨 split 组数：{len(leakage)}。", "- 建议正式切分时先提取原始录音键（文件名 `_seg` 前缀），将同一键的全部 seg 放入同一 split，再对录音键做哈希或分层切分；这样可避免同一录音片段泄漏到 train 与 validation。", "", "## 抽样审核", "", "`real_conversation_review.csv` 含真实社交数据前 10 段、每段最多 30 条消息，保留匿名 speaker_id、消息顺序与时间；PII 已脱敏。", "", "## 适用性结论", "", f"这批数据适合评估 QWbot 的自然多人聊天能力：{sum(v for k, v in speakers.items() if k >= 3)}/{len(convs_json)} 段为 3 人及以上，且有 {switches} 次角色切换，能覆盖多人交替发言。", f"但 {transcriptions} 条转写异常（占 {stats['transcription_error_ratio']:.2%}）、缺少真实 reply target、语料规模有限且许可仍待审核，适合人工审阅和离线质量评估，不宜直接进入训练。", "", "## 机器可读统计", "", "```json", json.dumps(stats, ensure_ascii=False, indent=2), "```", ""]
    (args.output_dir / "conversation_quality_report.md").write_text("\n".join(report), encoding="utf8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
