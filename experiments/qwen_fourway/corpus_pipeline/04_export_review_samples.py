#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, collections, csv, json, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from pipeline_common import FIELDS,csv_write

def main():
 p=argparse.ArgumentParser(description='Export auditable CSV for manual review; never writes formal training files.')
 p.add_argument('--check-only',action='store_true'); p.add_argument('--sample-posts',type=int,default=None); p.add_argument('--seed',type=int,default=42); p.add_argument('--output-dir',type=Path,default=Path('experiments/qwen_fourway/corpus_pipeline/output'))
 a=p.parse_args(); src=a.output_dir/'sft_candidates.jsonl'
 if not src.exists(): raise SystemExit(f'missing {src}; run 03_build_sft_candidates.py first')
 rows=[]
 with src.open(encoding='utf-8') as f:
  for line in f:
   try: rows.append(json.loads(line))
   except json.JSONDecodeError: pass
 if a.check_only: print(json.dumps({'rows':len(rows)},ensure_ascii=False)); return
 csv_write(a.output_dir/'review_samples.csv',rows,FIELDS)
 counts=collections.Counter(r.get('review_status') for r in rows); flags=collections.Counter()
 for r in rows: flags.update(x for x in (r.get('quality_flags') or '').split(',') if x)
 report={'rows':len(rows),'review_status_counts':dict(counts),'top_quality_flags':flags.most_common(20),'csv':str(a.output_dir/'review_samples.csv')}
 (a.output_dir/'review_export_stats.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8'); print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
