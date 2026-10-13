#!/usr/bin/env python3
"""Independent source, partition, and safety checks for V2 artifacts."""
import argparse, collections, csv, hashlib, importlib.util, json
from pathlib import Path

HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[3]
spec=importlib.util.spec_from_file_location('quality',HERE/'02_quality_grade.py'); quality=importlib.util.module_from_spec(spec); spec.loader.exec_module(quality)
def digest(s): return hashlib.sha256(s.encode('utf-8')).hexdigest()

def main():
    p=argparse.ArgumentParser(description='Validate V2 source traceability, count conservation and test isolation.')
    p.add_argument('--input-dir',type=Path,default=HERE/'output_full'); p.add_argument('--review-dir',type=Path,default=HERE/'output_review')
    p.add_argument('--check-only',action='store_true',help='report required file availability without scanning')
    a=p.parse_args()
    files=['style_candidates.jsonl','sft_candidates.jsonl','review_queue.jsonl','rejected.jsonl','duplicate_groups.jsonl','test_overlap_review.jsonl','quality_statistics.json','quality_report.md']
    missing=[name for name in files if not (a.input_dir/name).is_file()]
    if a.check_only:
        print(json.dumps({'missing':missing,'review_csv_exists':(a.review_dir/'stratified_human_review.csv').is_file()},ensure_ascii=False)); return
    if missing: p.error(f'missing output files: {missing}')
    stats=json.loads((a.input_dir/'quality_statistics.json').read_text(encoding='utf-8'))
    source_cache={}; ids=set(); counts=collections.Counter(); grades=collections.Counter(); splits={}; style_ids=set(); sft_ids=set(); review_ids=set()
    def source_row(row):
        name=row['source_file']
        if name not in source_cache:
            if source_cache: source_cache.clear()
            source_cache[name]=json.loads((ROOT/name).read_text(encoding='utf-8-sig'))
        post=source_cache[name][row['source_post_index']]
        assert post['回复列表'][row['reply_index']]==row['original_reply'], row['id']
        assert post['标题']==row['post_title_original'] and post['楼主内容']==row['op_body_original'],row['id']
        assert row['text_hash']==digest(quality.norm(row['original_reply']).casefold()),row['id']
        assert row['raw_text_hash']==digest(row['original_reply']),row['id']
    for filename in ('style_candidates.jsonl','rejected.jsonl'):
        with (a.input_dir/filename).open(encoding='utf-8') as f:
            for line in f:
                row=json.loads(line); rid=row['id']; assert rid not in ids,rid; ids.add(rid)
                counts[filename]+=1; grades[row['quality_grade']]+=1
                assert row['reply_relation']=='unknown' and row['license_status'].startswith('unconfirmed') and row['trainable'] is False
                assert row['review_status'] in ('pending_human_review','auto_quarantined')
                assert row['quality_grade'] in ('A','B','C','REJECT')
                assert not (row['data_split']=='test_quarantine' and filename=='style_candidates.jsonl')
                if filename=='style_candidates.jsonl':
                    style_ids.add(rid); assert not row['reject_reasons']
                    assert not quality.privacy_kinds(row['normalized_reply']+' '+row['post_title']+' '+row['op_body'])
                else: assert row['reject_reasons'] and row['quality_grade']=='REJECT'
                group=row['source_group_id']; split=row['data_split']
                if group in splits: assert splits[group]==split,group
                else: splits[group]=split
                source_row(row)
    for filename, target in [('sft_candidates.jsonl',sft_ids),('review_queue.jsonl',review_ids)]:
        with (a.input_dir/filename).open(encoding='utf-8') as f:
            for line in f:
                row=json.loads(line); rid=row['id']; assert rid in style_ids and rid not in target,rid
                target.add(rid); assert row['data_split']!='test_quarantine'
                if filename=='sft_candidates.jsonl': assert row['sft_candidate'] and row['semantic_evidence']['possible']
    assert review_ids==style_ids
    assert sft_ids <= style_ids
    assert counts['style_candidates.jsonl']+counts['rejected.jsonl']==stats['counts']['replies']
    assert counts['style_candidates.jsonl']==stats['counts']['style_candidates']
    assert len(sft_ids)==stats['counts']['sft_candidates']
    assert grades==collections.Counter(stats['grade_distribution'])
    assert stats['formal_trainable_count']==0
    test_path=ROOT/'datasets/QWbot_V44_真实群聊影子测试_匿名31题.jsonl'
    assert stats['test_sha256']==hashlib.sha256(test_path.read_bytes()).hexdigest()
    with (a.input_dir/'test_overlap_review.jsonl').open(encoding='utf-8') as f: test_rows=[json.loads(line) for line in f]
    assert len(test_rows)>=stats['counts'].get('test_overlap_replies',0)
    review_csv=a.review_dir/'stratified_human_review.csv'
    if review_csv.is_file():
        with review_csv.open(encoding='utf-8-sig',newline='') as f:
            for row in csv.DictReader(f):
                assert row['reviewer_decision']=='' and row['reviewer_notes']==''
                assert row['reply_relation']=='unknown'
                counts['review_csv_rows']+=1
    result={'valid':True,'source_replies_checked':len(ids),'style':len(style_ids),'sft':len(sft_ids),'review_queue':len(review_ids),'rejected':counts['rejected.jsonl'],'grades':dict(grades),'source_groups':len(splits),'test_records':len(test_rows),'review_csv_rows':counts['review_csv_rows']}
    print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__': main()
