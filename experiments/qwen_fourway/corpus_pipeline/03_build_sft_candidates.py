#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from pipeline_common import *
def main():
 p=argparse.ArgumentParser(description='Build isolated SFT review candidates and exclude exact/near matches to the 31-question test set.')
 p.add_argument('--check-only',action='store_true'); p.add_argument('--sample-posts',type=int,default=None); p.add_argument('--seed',type=int,default=42); p.add_argument('--output-dir',type=Path,default=Path('experiments/qwen_fourway/corpus_pipeline/output'))
 a=p.parse_args(); src=a.output_dir/'clean_candidates.jsonl'
 if not src.exists(): raise SystemExit(f'missing {src}; run 02_clean_kangba.py first')
 tests=test_texts(); exact={text_hash(x) for x in tests}; rows=[]; overlaps=[]
 for _,r in read_jsonl(src):
  text=norm(r.get('target_reply','')); compared=[text,norm(r.get('context','')),norm(r.get('incoming_message',''))]; exact_hit=any(text_hash(x) in exact for x in compared if x); near_hit=next((x for x in tests for y in compared if near(y,x)),None)
  r['quality_flags']=r.get('quality_flags','')+(',' if r.get('quality_flags') else '')+('test_exact_overlap' if exact_hit else '')+(',' if exact_hit and near_hit else '')+('test_near_duplicate' if near_hit and not exact_hit else '')
  if exact_hit or near_hit:
   r['review_status']='rejected_test_overlap' if exact_hit else 'pending_test_near_review'; overlaps.append({'id':r['id'],'source_reply_id':r['source_reply_id'],'target_reply':text,'match_type':'exact' if exact_hit else 'near','matched_test_text':text if exact_hit else near_hit})
  rows.append(r)
 if a.check_only: print(json.dumps({'candidates':len(rows),'test_texts':len(tests),'exact_or_near':len(overlaps)},ensure_ascii=False,indent=2)); return
 with (a.output_dir/'sft_candidates.jsonl').open('w',encoding='utf-8') as f:
  for r in rows: f.write(json.dumps(r,ensure_ascii=False)+'\n')
 (a.output_dir/'test_overlap_review.json').write_text(json.dumps({'test_count':len(tests),'overlaps':overlaps},ensure_ascii=False,indent=2),encoding='utf-8')
 print(json.dumps({'candidates':len(rows),'test_texts':len(tests),'exact_or_near':len(overlaps)},ensure_ascii=False))
if __name__=='__main__': main()
