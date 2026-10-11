#!/usr/bin/env python3
"""Parse selected IRC .raw/.ascii/.annotation files without inferring replies."""
from __future__ import annotations
import argparse, csv, hashlib, json, re, sys
from collections import defaultdict
from pathlib import Path
DEFAULT_RAW=Path("datasets/conversation_raw/irc_disentanglement")
DEFAULT_OUT=Path("experiments/qwen_fourway/conversation_pipeline/output_irc_pilot")

def split_line(line):
    return line.rstrip("\n").split("\t")

def read_messages(path):
    rows=[]; errors=[]
    for n,line in enumerate(path.read_text(encoding="utf-8",errors="replace").splitlines(),1):
        if not line.strip(): continue
        p=split_line(line)
        if len(p)==1: p=line.rstrip("\n").split(None, 3)
        # Canonical documented shape: message_id, timestamp, speaker, text.
        if len(p)>=4:
            mid, stamp, speaker, text=p[0],p[1],p[2],"\t".join(p[3:])
        elif len(p)==3:
            mid, speaker, text=p; stamp=None
        else:
            errors.append({"line":n,"reason":"too_few_tab_fields","raw":line}); continue
        if not mid.strip() or not speaker.strip(): errors.append({"line":n,"reason":"missing_id_or_speaker"}); continue
        rows.append({"raw_id":mid,"timestamp":stamp,"raw_speaker":speaker,"text":text,"source_line":n,"system":speaker.startswith("<") or speaker.upper() in {"SYSTEM","*"}})
    return rows,errors

def parse_annotations(path):
    links=[]; errors=[]
    for n,line in enumerate(path.read_text(encoding="utf-8",errors="replace").splitlines(),1):
        if not line.strip() or line.lstrip().startswith("#"): continue
        p=split_line(line)
        # Only explicit labelled edges are actionable. Generic numeric pairs are unresolved.
        if len(p)>=3 and p[0].lower() in {"reply","response","parent","context"}:
            links.append({"source":p[1],"target":p[2],"relation":p[0].lower(),"line":n})
        elif len(p)>=3 and p[2].lower() in {"reply","response","parent","context"}:
            links.append({"source":p[0],"target":p[1],"relation":p[2].lower(),"line":n})
        elif len(p)>=2:
            errors.append({"line":n,"reason":"unresolved_edge_direction_or_semantics","raw":line})
        else: errors.append({"line":n,"reason":"too_few_fields","raw":line})
    return links,errors

def parse(input_dir):
    files=sorted(p for p in input_dir.rglob("*") if p.is_file() and re.search(r"\.(raw|ascii|annotation)\.txt$",p.name))
    raw=[p for p in files if ".raw." in p.name]; ascii_files=[p for p in files if ".ascii." in p.name]
    logs=raw or ascii_files
    conversations=[]; all_messages=[]; errors=[]
    for path in logs:
        rows,errs=read_messages(path); errors += [{"source_file":str(path),**e} for e in errs]
        if not rows: continue
        cid="irc:"+hashlib.sha256(str(path.relative_to(input_dir)).encode()).hexdigest()[:16]
        speakers={}; by_id={}
        for order,r in enumerate(rows):
            key=r["raw_speaker"]; speakers.setdefault(key,f"speaker_{len(speakers)+1:04d}")
            mid=f"{cid}:m{order:06d}"; by_id[r["raw_id"]]=mid
            m={"conversation_id":cid,"message_id":mid,"speaker_id":speakers[key],"message_order":order,"text":r["text"],"reply_to_id":None,"reply_to_ids":[],"reply_relation":"unknown","source_log_id":cid,"source_channel":None,"source_file":str(path),"provenance_type":"irc_network_chat","license_status":"pending_confirmation","review_status":"pending","trainable":False,"system_message":r["system"]}
            all_messages.append(m)
        ann=next((p for p in files if ".annotation." in p.name and p.stem.split(".")[0]==path.stem.split(".")[0]),None)
        if ann:
            links,errs=parse_annotations(ann); errors += [{"source_file":str(ann),**e} for e in errs]
            targets=defaultdict(list)
            for link in links:
                if link["target"] in by_id and link["source"] in by_id: targets[link["target"]].append((by_id[link["source"]],link["relation"]))
                else: errors.append({"source_file":str(ann),"line":link["line"],"reason":"out_of_context_reference"})
            for m in all_messages[-len(rows):]:
                vals=targets.get(m["message_id"].split(":m",1)[-1],[]) # raw IDs are resolved below
                raw_target=next((k for k,v in by_id.items() if v==m["message_id"]),None)
                vals=targets.get(raw_target,[])
                m["reply_to_ids"]=[x[0] for x in vals]; m["reply_to_id"]=vals[0][0] if vals else None
                m["reply_relation"]="self_start" if any(x[0]==m["message_id"] for x in vals) else (vals[0][1] if len(vals)==1 else ("multiple" if vals else "unknown"))
        conversations.append({"conversation_id":cid,"message_count":len(rows),"speaker_count":len(speakers),"speaker_ids":list(speakers.values()),"source_group_id":cid,"split":"unassigned","source":"IRC Disentanglement","source_file":str(path),"provenance_type":"irc_network_chat","license_status":"pending_confirmation","review_status":"pending","trainable":False})
    return conversations,all_messages,errors

def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument("--input-dir",type=Path,default=DEFAULT_RAW); ap.add_argument("--output-dir",type=Path,default=DEFAULT_OUT); args=ap.parse_args(argv)
    if args.output_dir.exists() and any(args.output_dir.iterdir()): print(f"拒绝覆盖非空输出目录: {args.output_dir}",file=sys.stderr); return 4
    try: convs,msgs,errors=parse(args.input_dir)
    except (OSError,ValueError) as exc: print(f"IRC 解析失败: {exc}",file=sys.stderr); return 2
    args.output_dir.mkdir(parents=True,exist_ok=True)
    (args.output_dir/"conversations.jsonl").write_text("".join(json.dumps(x,ensure_ascii=False)+"\n" for x in convs),encoding="utf-8")
    (args.output_dir/"messages.jsonl").write_text("".join(json.dumps(x,ensure_ascii=False)+"\n" for x in msgs),encoding="utf-8")
    (args.output_dir/"parse_errors.json").write_text(json.dumps(errors,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(f"解析完成: {len(convs)} 个日志，{len(msgs)} 条消息，{len(errors)} 个解析/关联错误；全部 trainable=false")
    return 0
if __name__ == "__main__": raise SystemExit(main())
