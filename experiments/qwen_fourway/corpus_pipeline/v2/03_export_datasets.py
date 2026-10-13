#!/usr/bin/env python3
"""Export stratified manual review only; never emit training messages."""
import argparse, collections, csv, json, random
from pathlib import Path

HERE=Path(__file__).resolve().parent
FIELDS=['id','source_post_id','source_file','reply_index','post_title','op_body','candidate_text','quality_grade','quality_flags','quality_components','semantic_evidence','sft_candidate','sft_review_priority','reply_relation','review_status','license_status','data_split','original_reply_hash','text_quality','context_sufficiency','reply_target_verifiable','style_value','hallucination_risk','pii_status','reviewer_decision','reviewer_notes']

def main():
    p=argparse.ArgumentParser(description='Write deterministic UTF-8-SIG human review CSV; no training export.')
    p.add_argument('--input-dir',type=Path,default=HERE/'output_full')
    p.add_argument('--output-dir',type=Path,default=HERE/'output_review')
    p.add_argument('--per-grade',type=int,default=150)
    p.add_argument('--seed',type=int,default=20261009)
    p.add_argument('--check-only',action='store_true')
    a=p.parse_args()
    if a.per_grade <= 0: p.error('--per-grade must be positive')
    if not (a.input_dir/'quality_statistics.json').is_file(): p.error('missing full cleaner output')
    if a.check_only:
        print(json.dumps({'input_dir':str(a.input_dir),'review_dir_exists':a.output_dir.exists(),'per_grade':a.per_grade},ensure_ascii=False)); return
    if not a.output_dir.is_dir() or any(a.output_dir.iterdir()): p.error('review output directory must exist and be empty; use a new version path')
    rng=random.Random(a.seed); pools=collections.defaultdict(list); seen=collections.Counter()
    for name in ('style_candidates.jsonl','rejected.jsonl'):
        with (a.input_dir/name).open(encoding='utf-8') as f:
            for line in f:
                row=json.loads(line); grade=row['quality_grade']; seen[grade]+=1
                # Reservoir per quality grade. The sample retains high priority SFT candidates separately below.
                pool=pools[grade]
                if len(pool)<a.per_grade: pool.append(row)
                else:
                    j=rng.randrange(seen[grade])
                    if j<a.per_grade: pool[j]=row
    selected={r['id']:r for pool in pools.values() for r in pool}
    high=0
    with (a.input_dir/'sft_candidates.jsonl').open(encoding='utf-8') as f:
        for line in f:
            row=json.loads(line)
            if row['sft_review_priority']=='high' and high<a.per_grade:
                selected[row['id']]=row; high+=1
    with (a.output_dir/'stratified_human_review.csv').open('x',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=FIELDS); w.writeheader()
        for row in sorted(selected.values(),key=lambda r:(r['quality_grade'],r['id'])):
            out={k:'' for k in FIELDS}
            for key in ('id','source_post_id','source_file','reply_index','post_title','op_body','candidate_text','quality_grade','quality_flags','sft_candidate','sft_review_priority','reply_relation','review_status','license_status','data_split'): out[key]=row.get(key,'')
            for key in ('quality_components','semantic_evidence'): out[key]=json.dumps(row.get(key),ensure_ascii=False)
            out['original_reply_hash']=row['raw_text_hash']
            # CSV is redacted; raw source text remains in restricted JSONL quarantine.
            for key,value in out.items():
                if isinstance(value,str) and value.lstrip().startswith(('=','+','-','@')): out[key]="'"+value
            w.writerow(out)
    report={'sampled':len(selected),'per_grade_limit':a.per_grade,'grade_population':dict(seen),'grade_sample':dict(collections.Counter(r['quality_grade'] for r in selected.values())),'high_priority_sft_added':high,'seed':a.seed,'manual_fields_blank':FIELDS[-8:]}
    (a.output_dir/'sampling_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    for path in a.output_dir.iterdir(): path.chmod(0o600)
    print(json.dumps(report,ensure_ascii=False))
if __name__=='__main__': main()
