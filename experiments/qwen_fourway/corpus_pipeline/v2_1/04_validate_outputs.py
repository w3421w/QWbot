#!/usr/bin/env python3
"""Independent gates for V2.1 output artifacts."""
import argparse, collections, csv, hashlib, json, re, unicodedata
from difflib import SequenceMatcher
from pathlib import Path
RAW_FIELDS={'post_title_original','op_body_original','original_reply','source_url','normalized_reply'}
OUTPUTS=('style_candidates.jsonl','sft_candidates.jsonl','review_queue.jsonl','rejected.jsonl','test_quarantine.jsonl','duplicate_quarantine.jsonl')
def norm(s): return re.sub(r'\s+',' ',unicodedata.normalize('NFKC',s if isinstance(s,str) else '')).strip()
def compact(s): return re.sub(r'\W+','',norm(s).casefold())
def nhash(s): return hashlib.sha256(norm(s).casefold().encode('utf-8')).hexdigest()
def near(a,b):
    if min(len(a),len(b))<8 or max(len(a),len(b))>min(len(a),len(b))*1.18:return False
    return SequenceMatcher(None,a,b,autojunk=False).ratio()>=.90

def near_cross_split_pairs(train, validation):
    """Return every pair satisfying *this validator's* near predicate.

    The trigram inverted index is only a candidate generator; every candidate
    is still checked with ``near``.  Short texts also use character postings so
    that small replies with no shared trigram are not skipped.  A pair is
    yielded once, while all matches for either side are retained.
    """
    postings=collections.defaultdict(set)
    for j,(_, text) in enumerate(validation):
        grams=set(text[i:i+3] for i in range(len(text)-2))
        # Near comparisons require at least eight compact characters.  Use
        # trigrams for all normal candidates; character fallback is restricted
        # to very short values to avoid exploding postings on common Chinese
        # characters while still covering short replies.
        keys=grams if len(text)>=12 else set(text)
        for key in keys: postings[key].add(j)
    pairs=[]
    for i,(_, text) in enumerate(train):
        keys=set(text[k:k+3] for k in range(len(text)-2)) if len(text)>=12 else set(text)
        candidates=set()
        for key in keys: candidates.update(postings.get(key,()))
        for j in candidates:
            other=validation[j][1]
            if near(text,other): pairs.append((i,j))
    return pairs
def read(path):
    if not path.is_file(): return []
    return [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines() if x.strip()]
def main():
    p=argparse.ArgumentParser(description='Validate V2.1 output gates'); p.add_argument('--input-dir',type=Path,default=Path(__file__).parent/'output_full'); p.add_argument('--review-dir',type=Path,default=Path(__file__).parent/'output_review'); a=p.parse_args()
    stats=json.loads((a.input_dir/'quality_statistics.json').read_text(encoding='utf-8')); rows={fn:read(a.input_dir/fn) for fn in OUTPUTS}; test_ids={r['id'] for r in rows['test_quarantine.jsonl']}; ids=set(); splits=collections.defaultdict(set)
    for fn,items in rows.items():
      for r in items:
        assert not RAW_FIELDS.intersection(r),(fn,r.get('id')); assert r['reply_relation']=='unknown' and r['license_status'].startswith('unconfirmed') and r['trainable'] is False
        if fn not in ('sft_candidates.jsonl','review_queue.jsonl'): assert r['id'] not in ids,r['id']; ids.add(r['id'])
        splits[r['source_group_id']].add(r['data_split'])
    for fn in ('style_candidates.jsonl','sft_candidates.jsonl','review_queue.jsonl','rejected.jsonl'): assert test_ids.isdisjoint({r['id'] for r in rows[fn]}),('test_leak',fn)
    assert all(len({x for x in ss if x in ('train','validation')})<=1 for ss in splits.values())
    retained=[r for r in rows['style_candidates.jsonl'] if r['data_split'] in ('train','validation')]; by_split={s:[r for r in retained if r['data_split']==s] for s in ('train','validation')}
    exact={}
    for s,items in by_split.items():
      for r in items:
        h=nhash(r.get('candidate_text','')); assert h not in exact or exact[h][0]==s,('exact_cross_split',h,exact[h][0],s); exact[h]=(s,r['id'])
    train_near=[(r['id'],compact(r.get('candidate_text',''))) for r in by_split['train']]
    validation_near=[(r['id'],compact(r.get('candidate_text',''))) for r in by_split['validation']]
    near_hits=[(train_near[i][0],validation_near[j][0]) for i,j in near_cross_split_pairs(train_near,validation_near)]
    assert not near_hits,('near_cross_split',near_hits[:5])
    final={k:len(rows[f'{k}.jsonl']) for k in ('style_candidates','sft_candidates','rejected','test_quarantine','duplicate_quarantine')}; assert {k:stats['final_candidates'][k] for k in final}==final,(stats['final_candidates'],final)
    total=final['style_candidates']+final['rejected']+final['test_quarantine']+final['duplicate_quarantine']; assert total==stats['original_reply_total']==stats['counts']['replies'],(total,stats['original_reply_total'],stats['counts']['replies'])
    for s in ('train','validation'):
      assert stats['final_candidates'][f'{s}_replies']==len(by_split[s]); assert stats['final_candidates'][f'{s}_posts']==len({r['source_post_id'] for r in by_split[s]})
    assert stats['conservation_check']['passed'] is True and stats['formal_trainable_count']==0
    if (a.review_dir/'stratified_human_review.csv').is_file():
      with (a.review_dir/'stratified_human_review.csv').open(encoding='utf-8-sig',newline='') as f: assert not RAW_FIELDS.intersection(set(next(csv.reader(f))))
    print(json.dumps({'valid':True,'final_candidates':final,'original_reply_total':stats['original_reply_total'],'train_replies':len(by_split['train']),'validation_replies':len(by_split['validation']),'near_cross_split':0},ensure_ascii=False))
if __name__=='__main__': main()
