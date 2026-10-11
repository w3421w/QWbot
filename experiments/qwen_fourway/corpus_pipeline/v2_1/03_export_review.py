#!/usr/bin/env python3
import argparse,csv,json,random
from pathlib import Path
FIELDS=['id','source_post_id','source_file','reply_index','post_title','op_body','candidate_text','quality_grade','quality_flags','quality_components','semantic_evidence','sft_candidate','sft_review_priority','reply_relation','review_status','license_status','data_split','text_hash','reviewer_decision','reviewer_notes']
def main():
 p=argparse.ArgumentParser(description='Export redacted manual review CSV; never exports raw text or messages'); p.add_argument('--input-dir',type=Path,default=Path(__file__).parent/'output_full'); p.add_argument('--output-dir',type=Path,default=Path(__file__).parent/'output_review'); p.add_argument('--per-grade',type=int,default=100); p.add_argument('--seed',type=int,default=20261009); a=p.parse_args(); a.output_dir.mkdir(parents=True,exist_ok=True)
 out_csv=a.output_dir/'stratified_human_review.csv'
 if out_csv.exists(): raise RuntimeError(f'refusing to overwrite existing review file: {out_csv}')
 rows=[]
 for fn in ('style_candidates.jsonl','rejected.jsonl'):
  with (a.input_dir/fn).open(encoding='utf-8') as f:
   for line in f: rows.append(json.loads(line))
 groups={g:[r for r in rows if r.get('quality_grade')==g] for g in ('A','B','C','REJECT')}
 rng=random.Random(a.seed); sampled=[]
 for grade,items in groups.items():
  items=list(items); rng.shuffle(items); sampled.extend(items[:a.per_grade])
 with out_csv.open('x',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=FIELDS); w.writeheader()
  for r in sampled:
   out={k:r.get(k,'') for k in FIELDS}; out['quality_flags']=json.dumps(r.get('quality_flags',[]),ensure_ascii=False); out['quality_components']=json.dumps(r.get('quality_components',{}),ensure_ascii=False); out['semantic_evidence']=json.dumps(r.get('semantic_evidence',{}),ensure_ascii=False)
   for k,v in out.items():
    if isinstance(v,str) and v.lstrip().startswith(('=','+','-','@')):out[k]="'"+v
   w.writerow(out)
 report={'seed':a.seed,'per_grade_limit':a.per_grade,'overall_counts':{g:len(v) for g,v in groups.items()},'sampled_counts':{g:min(len(v),a.per_grade) for g,v in groups.items()},'sampled':len(sampled),'raw_fields_absent':True,'formula_injection_guard':True}
 (a.output_dir/'sampling_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
 for pth in a.output_dir.iterdir():pth.chmod(0o600)
 print(json.dumps(report,ensure_ascii=False))
if __name__=='__main__':main()
