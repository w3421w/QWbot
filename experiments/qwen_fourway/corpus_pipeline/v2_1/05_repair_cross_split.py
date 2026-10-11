#!/usr/bin/env python3
"""Repair missed train/validation near duplicates in an existing V2.1 output.

Only the already exported, redacted artifacts are read.  Validation rows that
conflict with one or more train rows are moved to a new quarantine directory;
the original output is never modified.
"""
from __future__ import annotations
import argparse, copy, hashlib, json, time
from difflib import SequenceMatcher
from pathlib import Path
import importlib.util

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('validator', HERE/'04_validate_outputs.py')
validator=importlib.util.module_from_spec(spec); spec.loader.exec_module(validator)

FILES=('style_candidates.jsonl','sft_candidates.jsonl','review_queue.jsonl',
       'rejected.jsonl','test_quarantine.jsonl','duplicate_quarantine.jsonl')

def read(path):
    return [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines() if x.strip()]

def write(path, rows):
    with path.open('x', encoding='utf-8') as f:
        for row in rows: f.write(json.dumps(row,ensure_ascii=False,separators=(',',':'))+'\n')

def main():
    p=argparse.ArgumentParser(description='Repair existing V2.1 cross-split near duplicates')
    p.add_argument('--input-dir',type=Path,default=HERE/'output_full')
    p.add_argument('--output-dir',type=Path,default=HERE/'output_full_repaired_01')
    p.add_argument('--review-dir',type=Path,default=HERE/'output_review_repaired_01')
    a=p.parse_args()
    if a.output_dir.exists() or a.review_dir.exists():
        raise RuntimeError('refusing to overwrite existing repair output or review directory')
    required=('quality_statistics.json','style_candidates.jsonl','sft_candidates.jsonl','review_queue.jsonl',
              'rejected.jsonl','test_quarantine.jsonl','duplicate_quarantine.jsonl')
    for name in required:
        if not (a.input_dir/name).is_file(): raise FileNotFoundError(a.input_dir/name)
    started=time.perf_counter()
    rows={name:read(a.input_dir/name) for name in FILES}
    retained=[r for r in rows['style_candidates.jsonl'] if r.get('data_split') in ('train','validation')]
    train=[(r['id'],validator.compact(r.get('candidate_text',''))) for r in retained if r['data_split']=='train']
    validation=[(r['id'],validator.compact(r.get('candidate_text',''))) for r in retained if r['data_split']=='validation']
    pairs=validator.near_cross_split_pairs(train,validation)
    conflicts={}
    for ti,vi in pairs:
        tid,ttext=train[ti]; vid,vtext=validation[vi]
        conflicts.setdefault(vid,[]).append({'train_id':tid,'similarity':SequenceMatcher(None,ttext,vtext,autojunk=False).ratio()})
    by_id={r['id']:r for r in rows['style_candidates.jsonl']}
    moved=[]
    for vid, evidence in conflicts.items():
        r=copy.deepcopy(by_id[vid]); r['data_split']='duplicate_quarantine'
        r.setdefault('reject_reasons',[]).append('cross_split_duplicate_isolation_repair_01')
        r['near_duplicate_conflicts']=sorted(evidence,key=lambda x:x['train_id'])
        r['near_duplicate_of']=evidence[0]['train_id']
        r['sft_candidate']=False; r['sft_review_priority']=None
        moved.append(r)
    moved_ids={r['id'] for r in moved}
    # Keep all train rows, and remove repaired validation rows from every
    # downstream candidate view.  Review queue and SFT therefore remain exact
    # projections of the repaired style candidates.
    rows['style_candidates.jsonl']=[r for r in rows['style_candidates.jsonl'] if r['id'] not in moved_ids]
    rows['sft_candidates.jsonl']=[r for r in rows['sft_candidates.jsonl'] if r['id'] not in moved_ids]
    rows['review_queue.jsonl']=[r for r in rows['review_queue.jsonl'] if r['id'] not in moved_ids]
    rows['duplicate_quarantine.jsonl'].extend(moved)
    a.output_dir.mkdir(parents=True,mode=0o700); a.review_dir.mkdir(parents=True,mode=0o700)
    for name in FILES: write(a.output_dir/name,rows[name])
    # Preserve duplicate groups, audit and test review evidence byte-for-byte.
    for name in ('duplicate_groups.jsonl','internal_audit_restricted.jsonl','test_overlap_review.jsonl'):
        src=a.input_dir/name
        if src.is_file(): (a.output_dir/name).write_bytes(src.read_bytes())
    old=json.loads((a.input_dir/'quality_statistics.json').read_text(encoding='utf-8'))
    stats=copy.deepcopy(old); final={k:len(rows[f'{k}.jsonl']) for k in ('style_candidates','sft_candidates','rejected','test_quarantine','duplicate_quarantine')}
    style=rows['style_candidates.jsonl']; retained2=[r for r in style if r.get('data_split') in ('train','validation')]
    final.update({'train_replies':sum(r['data_split']=='train' for r in retained2),'validation_replies':sum(r['data_split']=='validation' for r in retained2),'train_posts':len({r['source_post_id'] for r in retained2 if r['data_split']=='train'}),'validation_posts':len({r['source_post_id'] for r in retained2 if r['data_split']=='validation'})})
    stats['final_candidates']=final; stats['counts']['style_candidates']=final['style_candidates']; stats['counts']['sft_candidates']=final['sft_candidates']; stats['counts']['review_queue']=len(rows['review_queue.jsonl']); stats['counts']['duplicate_quarantined_replies']=final['duplicate_quarantine']; stats['counts']['cross_split_near_repair_isolated']=len(moved)
    stats['split_counts']['train']['replies']=final['train_replies']; stats['split_counts']['validation']['replies']=final['validation_replies']; stats['split_counts'].setdefault('duplicate_quarantine',{})['replies']=final['duplicate_quarantine']
    total=sum(final[k] for k in ('style_candidates','rejected','test_quarantine','duplicate_quarantine'))
    stats['conservation_check']={'original_replies':stats['original_reply_total'],'final_output_rows':total,'passed':total==stats['original_reply_total']}
    stats['repair']={'script':'05_repair_cross_split.py','version':'01','source_output':str(a.input_dir),'isolated_validation_rows':len(moved),'near_cross_split_pairs':len(pairs),'near_cross_split_validation_ids':len(conflicts),'elapsed_seconds':round(time.perf_counter()-started,3),'all_conflict_evidence_retained':True}
    (a.output_dir/'quality_statistics.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2),encoding='utf-8')
    (a.output_dir/'quality_report.md').write_text('# V2.1 输出修复报告\n\n'+json.dumps(stats,ensure_ascii=False,indent=2),encoding='utf-8')
    for path in a.output_dir.iterdir(): path.chmod(0o600)
    print(json.dumps({'near_cross_split_pairs':len(pairs),'isolated_validation_rows':len(moved),'final_candidates':final,'conservation_passed':stats['conservation_check']['passed']},ensure_ascii=False))

if __name__=='__main__': main()
