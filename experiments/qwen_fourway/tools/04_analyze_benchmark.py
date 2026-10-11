#!/usr/bin/env python3
"""离线分析四模型真实群聊测试结果，并生成可复现的人工盲评表。"""
import argparse
import csv
import hashlib
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT.parents[1] / "datasets" / "QWbot_V44_真实群聊影子测试_匿名31题.jsonl"
REPORT_DIR = ROOT / "reports"
CASES = {
    "01": "01_1p7b_non_thinking",
    "02": "02_1p7b_thinking",
    "03": "03_4b_non_thinking",
    "04": "04_4b_thinking",
}
MODEL_LABELS = {
    "01": "01_1p7b_non_thinking",
    "02": "02_1p7b_thinking",
    "03": "03_4b_non_thinking",
    "04": "04_4b_thinking",
}
LETTERS = ("A", "B", "C", "D")


def parse_args():
    ap = argparse.ArgumentParser(description="分析四模型 benchmark_real_chat JSONL，生成离线统计和人工盲评 CSV。")
    ap.add_argument("--tag", required=True, help="结果文件标签，例如 test5_20261009")
    ap.add_argument("--seed", type=int, default=20261009, help="盲映射随机种子（默认：20261009）")
    ap.add_argument("--data", type=Path, default=DATA_PATH, help="匿名题集 JSONL 路径")
    ap.add_argument("--reports-dir", type=Path, default=REPORT_DIR, help="报告输出目录")
    return ap.parse_args()


def read_jsonl(path):
    rows, malformed = [], []
    if not path.is_file():
        return rows, [f"文件不存在: {path}"]
    with path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    rows.append(obj)
                else:
                    malformed.append(f"第{line_no}行不是对象")
            except json.JSONDecodeError as exc:
                malformed.append(f"第{line_no}行 JSON 错误: {exc}")
    return rows, malformed


def load_questions(path):
    rows, malformed = read_jsonl(path)
    questions = {}
    issues = list(malformed)
    for row in rows:
        qid = row.get("id")
        if not isinstance(qid, str) or not qid:
            issues.append("题集存在缺少有效 id 的记录")
            continue
        if qid in questions:
            issues.append(f"题集重复 id: {qid}")
        questions[qid] = row
    return questions, issues


def result_path(case, tag):
    return ROOT / CASES[case] / "results" / f"benchmark_real_chat_{tag}.jsonl"


def key_of(row):
    qid, repeat = row.get("question_id"), row.get("repeat")
    return (qid, repeat) if isinstance(qid, str) and qid and type(repeat) is int and repeat >= 1 else None


def is_incomplete(row):
    err = str(row.get("error") or "").lower()
    return err.startswith("incomplete") or (row.get("response_ok") is False and row.get("thinking") is True
                                             and row.get("thinking_closed") is False)


def is_error(row):
    err = str(row.get("error") or "").strip()
    return not is_incomplete(row) and (bool(err) or row.get("response_ok") is not True)


def record_problems(row, case, questions):
    problems = []
    key = key_of(row)
    if key is None:
        problems.append("question_id/repeat 无效")
    elif key[0] not in questions:
        problems.append("question_id 不在题集")
    if row.get("case") != case:
        problems.append("case 与结果目录不一致")
    if not isinstance(row.get("response_ok"), bool):
        problems.append("response_ok 不是布尔值")
    if not isinstance(row.get("answer"), str):
        problems.append("answer 不是字符串")
    if row.get("response_ok") is True and (not row.get("answer") or row.get("error")):
        problems.append("有效回复状态与 answer/error 冲突")
    for field in ("generated_tokens", "generate_seconds", "peak_vram_gb"):
        value = row.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
            problems.append(f"{field} 无效")
    return problems


def average(rows, field):
    vals = []
    for row in rows:
        value = row.get(field)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            vals.append(float(value))
    return round(sum(vals) / len(vals), 3) if vals else None


def model_statistics(rows, expected_keys, case, questions):
    duplicates = Counter(k for k in (key_of(r) for r in rows) if k is not None)
    unique = {k: r for r in rows if (k := key_of(r)) is not None}
    missing = sorted(expected_keys - set(unique), key=lambda x: (str(x[0]), x[1]))
    invalid_key_records = sum(key_of(r) is None for r in rows)
    incomplete = sum(is_incomplete(r) for r in rows)
    errors = sum(is_error(r) or bool(record_problems(r, case, questions)) for r in rows)
    return {
        "total_test_count": len(rows),
        "unique_test_count": len(unique),
        "valid_response_count": sum(r.get("response_ok") is True and not is_incomplete(r)
                                    and not record_problems(r, case, questions) for r in rows),
        "incomplete_count": incomplete,
        "error_count": errors,
        "hit_token_limit_count": sum(bool(r.get("hit_token_limit")) for r in rows),
        "average_generate_seconds": average(rows, "generate_seconds"),
        "average_generated_tokens": average(rows, "generated_tokens"),
        "average_peak_vram_gb": average(rows, "peak_vram_gb"),
        "duplicate_records": sum(n - 1 for n in duplicates.values() if n > 1),
        "duplicate_keys": [list(k) for k, n in sorted(duplicates.items()) if n > 1],
        "missing_keys": [list(k) for k in missing],
        "invalid_key_records": invalid_key_records,
    }


def stable_rng(seed, key):
    digest = hashlib.sha256(f"{seed}:{key[0]}:{key[1]}".encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def context_text(context):
    if not context:
        return ""
    return "\n".join(f"{item.get('speaker', '')}：{item.get('text', '')}" for item in context)


def safe_reply(row):
    if not row:
        return ""
    if is_incomplete(row):
        return "未完成"
    if is_error(row):
        return "错误"
    return str(row.get("answer") or "")


def build_blind(questions, records, expected_keys, seed):
    blind_rows, blind_key = [], {}
    for key in sorted(expected_keys, key=lambda x: (str(x[0]), x[1])):
        qid, repeat = key
        rng = stable_rng(seed, key)
        shuffled = list(CASES)
        rng.shuffle(shuffled)
        mapping = dict(zip(LETTERS, shuffled))
        blind_key[f"{qid}::{repeat}"] = {letter: case for letter, case in mapping.items()}
        question = questions.get(qid, {})
        out = {
            "question_id": qid, "repeat": repeat,
            "context": context_text(question.get("context", [])),
            "incoming_message": str(question.get("incoming_message", "")),
        }
        by_case = {case: records.get(case, {}).get(key) for case in CASES}
        for letter in LETTERS:
            row = by_case.get(mapping[letter])
            out[f"{letter}_final_reply"] = safe_reply(row)
            out[f"{letter}_response_status"] = ("有效" if row and not is_incomplete(row) and not is_error(row)
                                                  else "未完成" if row and is_incomplete(row)
                                                  else "错误" if row else "缺失")
            for metric in ("context_understanding", "naturalness", "anti_ba_style", "factual_risk", "review_notes"):
                out[f"{letter}_{metric}"] = ""
        blind_rows.append(out)
    return blind_rows, blind_key


def write_csv(path, rows):
    fields = list(rows[0]) if rows else ["question_id", "repeat", "context", "incoming_message"]
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_report(path, tag, seed, stats, alignment, blind_rows, issues):
    lines = [f"# 四模型实验分析：{tag}", "", f"- 盲映射 seed：`{seed}`", f"- 盲评行数：{len(blind_rows)}", "", "## 对齐与数据质量", ""]
    lines += [f"- 对齐测试键数：{alignment['expected_key_count']}", f"- 重复记录数（合计）：{alignment['duplicate_records']}",
              f"- 缺失记录数（合计）：{alignment['missing_records']}", f"- 错误/格式问题：{len(issues)}"]
    if issues:
        lines += ["", "```text"] + [str(x) for x in issues] + ["```"]
    lines += ["", "## 模型统计", "", "| 模型 |总测试|有效|未完成|错误|命中上限|平均秒|平均生成Token|平均峰值显存GB|", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for case, label in MODEL_LABELS.items():
        s = stats[case]
        lines.append(f"| {label} | {s['total_test_count']} | {s['valid_response_count']} | {s['incomplete_count']} | {s['error_count']} | {s['hit_token_limit_count']} | {s['average_generate_seconds']} | {s['average_generated_tokens']} | {s['average_peak_vram_gb']} |")
    lines += ["", "## 历史上下文分组", ""]
    for group in ("with_history", "without_history"):
        lines.append(f"### {group}")
        for case, label in MODEL_LABELS.items():
            lines.append(f"- {label}: {stats['by_context'][group][case]}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    if not re.fullmatch(r"[A-Za-z0-9._-]+", args.tag):
        raise SystemExit("--tag只能包含字母、数字、点、下划线和连字符")
    prefix = f"four_model_{args.tag}"
    paths = {"blind_scoring": args.reports_dir / f"{prefix}_blind_scoring.csv",
             "blind_key": args.reports_dir / f"{prefix}_blind_key.json",
             "statistics": args.reports_dir / f"{prefix}_statistics.json",
             "report": args.reports_dir / f"{prefix}_report.md"}
    conflicts = [str(p) for p in paths.values() if p.exists()]
    if conflicts:
        raise SystemExit("输出文件已存在，不会覆盖；请更换 --tag:\n" + "\n".join(conflicts))
    questions, issues = load_questions(args.data)
    records, rows_by_case, raw_counts = {}, {}, {}
    for case in CASES:
        rows, malformed = read_jsonl(result_path(case, args.tag))
        records[case] = {}
        rows_by_case[case] = []
        raw_counts[case] = len(rows)
        issues.extend(f"{case}: {x}" for x in malformed)
        for row in rows:
            rows_by_case[case].append(row)
            for problem in record_problems(row, case, questions):
                issues.append(f"{case}: {row.get('question_id', '?')} repeat={row.get('repeat', '?')}: {problem}")
            key = key_of(row)
            if key is None:
                continue
            if key in records[case]:
                issues.append(f"{case}: 重复记录 {key[0]} repeat={key[1]}")
            records[case][key] = row
    expected = set().union(*(set(records[c]) for c in CASES))
    if not expected:
        issues.append("四组结果均无可对齐记录")
    stats = {case: model_statistics(rows_by_case[case], expected, case, questions) for case in CASES}
    context_groups = {"with_history": [], "without_history": []}
    for key in expected:
        context_groups["with_history" if questions.get(key[0], {}).get("context") else "without_history"].append(key)
    stats["by_context"] = {}
    for group, keys in context_groups.items():
        stats["by_context"][group] = {case: model_statistics([records[case][k] for k in keys if k in records[case]], set(keys), case, questions) for case in CASES}
    blind_rows, blind_key = build_blind(questions, records, expected, args.seed)
    alignment = {"expected_key_count": len(expected), "duplicate_records": sum(stats[c]["duplicate_records"] for c in CASES),
                 "missing_records": sum(len(stats[c]["missing_keys"]) for c in CASES), "raw_record_counts": raw_counts}
    payload = {"tag": args.tag, "seed": args.seed, "expected_key_count": len(expected), "alignment": alignment,
               "models": {case: stats[case] for case in CASES}, "by_context": stats["by_context"], "issues": issues}
    args.reports_dir.mkdir(parents=True, exist_ok=True)
    write_csv(paths["blind_scoring"], blind_rows)
    paths["blind_key"].write_text(json.dumps({"tag": args.tag, "seed": args.seed, "mapping": blind_key}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    paths["statistics"].write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_report(paths["report"], args.tag, args.seed, stats, alignment, blind_rows, issues)
    print(json.dumps({"tag": args.tag, "rows": len(blind_rows), "outputs": {k: str(v) for k, v in paths.items()}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
