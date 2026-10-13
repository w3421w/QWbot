#!/usr/bin/env python3
"""Audit parsed IRC JSONL, including real parse/association error counts."""
from __future__ import annotations
import argparse,csv,hashlib,json,re,sys,unicodedata
from collections import Counter,defaultdict
from pathlib import Path
DEFAULT_OUT=Path("experiments/qwen_fourway/conversation_pipeline/output_irc_pilot")
DEFAULT_TEST=Path("datasets/QWbot_V44_真实群聊影子测试_匿名31题.jsonl")
PII=re.compile(r"1[3-9]\d{9}|[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}|\b\d{17}[0-9Xx]\b")
def read(p): return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
def norm(x): return re.sub(r"\s+","",unicodedata.normalize("NFKC",str(x or ""))).casefold()
def audit(convs,msgs,errors,tests):
    by=defaultdict(list)
    for m in msgs: by[m["conversation_id"]].append(m)
    issues=[]; ids={c["conversation_id"] for c in convs}; seen=set()
    for c in convs:
        rows=by.get(c["conversation_id"],[])
        if c.get("message_count")!=len(rows): issues.append(f"{c['conversation_id']}: count mismatch")
        if [m.get("message_order") for m in rows]!=list(range(len(rows))): issues.append(f"{c['conversation_id']}: order mismatch")
        if c.get("trainable") is not False: issues.append(f"{c['conversation_id']}: trainable")
        for m in rows:
            if m.get("message_id") in seen: issues.append("duplicate message_id")
            seen.add(m.get("message_id"))
            if m.get("trainable") is not False: issues.append("message trainable")
            targets=m.get("reply_to_ids") or ([] if m.get("reply_to_id") is None else [m.get("reply_to_id")])
            if any(t not in seen and t!=m.get("message_id") for t in targets): issues.append(f"{m['message_id']}: target outside prior/context")
    speakers=Counter(len({m["speaker_id"] for m in by[c["conversation_id"]]}) for c in convs)
    replies=[m for m in msgs if m.get("reply_to_ids")]; multi=[m for m in msgs if len(m.get("reply_to_ids",[]))>1]; starts=[m for m in msgs if m.get("reply_relation")=="self_start"]
    chinese=sum(bool(re.search(r"[\u3400-\u9fff]",m.get("text",""))) for m in msgs)
    return {"conversation_count":len(convs),"message_count":len(msgs),"distinct_speaker_count":len({m["speaker_id"] for m in msgs}),"reply_relation_count":sum(len(m.get("reply_to_ids",[])) for m in msgs),"messages_with_reply":len(replies),"multi_association_message_count":len(multi),"conversation_start_count":len(starts),"unresolved_association_count":sum(1 for e in errors if "unresolved" in e.get("reason","") or "out_of_context" in e.get("reason","")),"parse_error_count":len(errors),"parse_error_types":dict(Counter(e.get("reason","unknown") for e in errors)),"branching_conversation_count":sum(any(len(m.get("reply_to_ids",[]))>1 for m in by[c["conversation_id"]]) for c in convs),"duplicate_message_count":len(msgs)-len(seen),"privacy_risk_message_count":sum(bool(PII.search(str(m.get("text","")))) for m in msgs),"chinese_message_ratio":chinese/len(msgs) if msgs else 0,"license_status":"pending_confirmation","trainable_true_count":sum(x.get("trainable") is True for x in convs+msgs),"quantity_conservation":sum(c.get("message_count",0) for c in convs)==len(msgs),"speaker_count_distribution":dict(sorted((str(k),v) for k,v in speakers.items())),"test_set_question_count":len(tests),"test_overlap_exact":sum(1 for m in msgs for t in tests if norm(m.get("text")) and norm(m.get("text")) in norm(t.get("incoming_message", ""))),"validation_errors":issues}
def main(argv=None):
    ap=argparse.ArgumentParser(); ap.add_argument("--output-dir",type=Path,default=DEFAULT_OUT); ap.add_argument("--test-set",type=Path,default=DEFAULT_TEST); args=ap.parse_args(argv); out=args.output_dir
    targets=[out/"quality_statistics.json",out/"real_conversation_review.csv",out/"license_review.md",out/"pilot_report.md"]
    if any(p.exists() for p in targets): print("拒绝覆盖已有审计文件",file=sys.stderr); return 4
    try:
        convs,msgs=read(out/"conversations.jsonl"),read(out/"messages.jsonl"); errors=json.loads((out/"parse_errors.json").read_text(encoding="utf-8")) if (out/"parse_errors.json").exists() else []
        tests=read(args.test_set) if args.test_set.exists() else []
        stats=audit(convs,msgs,errors,tests)
    except (OSError,ValueError,KeyError,json.JSONDecodeError) as exc: print(f"审计失败: {exc}",file=sys.stderr); return 2
    out.mkdir(parents=True,exist_ok=True); (out/"quality_statistics.json").write_text(json.dumps(stats,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    with (out/"real_conversation_review.csv").open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.writer(f); w.writerow(["conversation_id","message_id","speaker_id","message_order","text_redacted","reply_count","review_status","privacy_redacted"])
        for m in msgs[:200]:
            raw=str(m.get("text","")); clean=PII.sub("[已脱敏]",raw); w.writerow([m["conversation_id"],m["message_id"],m["speaker_id"],m["message_order"],clean,len(m.get("reply_to_ids",[])),m.get("review_status"),clean!=raw])
    (out/"license_review.md").write_text("# IRC Disentanglement 许可审核\n\n来源为官方 `jkkummerfeld/irc-disentanglement`。训练与再分发许可待人工确认；所有记录 `trainable=false`。\n",encoding="utf-8")
    (out/"pilot_report.md").write_text("# IRC 真实群聊试点审计\n\n"+"\n".join(f"- {k}: {v}" for k,v in stats.items() if k not in {"validation_errors"})+"\n",encoding="utf-8")
    print(json.dumps(stats,ensure_ascii=False,indent=2)); return 3 if stats["validation_errors"] else 0
if __name__=="__main__": raise SystemExit(main())
