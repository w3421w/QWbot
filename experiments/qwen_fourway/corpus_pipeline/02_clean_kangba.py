#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, collections, hashlib, json, random, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from pipeline_common import *

def iter_posts(sample_posts, seed):
 files=sorted(RAW_DIR.rglob('tieba_post_detail_page_*.json')); rng=random.Random(seed); reservoir=[]; seen=0
 for path in files:
  try: data=json.loads(path.read_text(encoding='utf-8'))
  except Exception: continue
  if not isinstance(data,list): continue
  for ix,item in enumerate(data):
   if not isinstance(item,dict): continue
   rec=(path,ix,item); seen+=1
   if sample_posts is None: yield rec
   elif len(reservoir)<sample_posts: reservoir.append(rec)
   else:
    j=rng.randrange(seen)
    if j<sample_posts: reservoir[j]=rec
 if sample_posts is not None:
  for rec in reservoir: yield rec

def main():
 p=argparse.ArgumentParser(description='Clean raw Kangba records into auditable candidates; raw files remain read-only.')
 p.add_argument('--check-only',action='store_true'); p.add_argument('--sample-posts',type=int,default=None,help='deterministic reservoir sample of posts'); p.add_argument('--seed',type=int,default=42); p.add_argument('--output-dir',type=Path,default=Path('experiments/qwen_fourway/corpus_pipeline/output'))
 a=p.parse_args(); stats=collections.Counter(); exact=set(); bucket={}; candidates=[]; rejects=[]; relation_cases=[]; posts=set(); replies=0
 for path,ix,item in iter_posts(a.sample_posts,a.seed):
  posts.add((str(path),ix)); pid=post_id(norm(item.get('链接')),str(path),ix); title=redact(norm(item.get('标题'))); body=redact(norm(item.get('楼主内容'))); context='\n'.join(x for x in (title,body) if x and x!='[无法获取楼主内容]') or '[贴吧讨论帖]'; stats['posts_with_body' if body and body!='[无法获取楼主内容]' else 'posts_without_body']+=1; rs=item.get('回复列表',[])
  if not isinstance(rs,list): stats['invalid_reply_list']+=1; continue
  for ri,raw in enumerate(rs):
   replies+=1; original=norm(raw); text=redact(original); reason=classify(text); h=text_hash(text); flags=['reply_relation_unknown','author_missing','timestamp_missing','quote_missing','license_unconfirmed']
   if text==original and any(x in text for x in ('手机号','微信','QQ')): flags.append('privacy_review')
   if h in exact: flags.append('exact_duplicate')
   exact.add(h)
   b=(len(text)//10, key(text)[:24]); near_id='';
   for oldh,oldtext,oldid in bucket.get(b,[]):
    if near(text,oldtext): near_id=oldid; flags.append('near_duplicate_risk'); break
   bucket.setdefault(b,[]).append((h,text,f'{pid}:{ri}'))
   rid='reply-'+hashlib.sha1(f'{pid}:{ri}:{text}'.encode()).hexdigest()[:16]
   row={'id':hashlib.sha1(f'{pid}:{rid}'.encode()).hexdigest()[:20],'source':'tieba_kangba_raw','conversation_id':f'post-{pid}','context':context,'incoming_message':context,'target_reply':text,'source_post_id':pid,'source_reply_id':rid,'source_file':str(path.relative_to(ROOT)),'reply_index':ri,'reply_relation':'unknown','quality_flags':','.join(flags),'quality_score':json.dumps({'reply_relation':0.0,'text_naturalness':natural_score(text)},ensure_ascii=False,separators=(',',':')),'provenance_type':'real_reply','review_status':'rejected' if reason else 'pending_review','license_status':'unconfirmed','text_hash':h,'near_duplicate_of':near_id}
   if reason:
    row['quality_flags'] += ','+reason; rejects.append({**row,'reject_reason':reason}); stats['rejected_'+reason]+=1
   else: candidates.append(row); relation_cases.append(row); stats['pending_review']+=1
 stats.update(posts=len(posts),replies_seen=replies,clean_candidates=len(candidates),rejected=len(rejects))
 a.output_dir.mkdir(parents=True,exist_ok=True)
 if a.check_only:
  print(json.dumps(dict(stats),ensure_ascii=False,indent=2)); return
 for name,rows in [('clean_candidates.jsonl',candidates),('rejected.jsonl',rejects),('relation_uncertain.jsonl',relation_cases)]:
  with (a.output_dir/name).open('w',encoding='utf-8') as f:
   for row in rows: f.write(json.dumps(row,ensure_ascii=False)+'\n')
 (a.output_dir/'cleaning_stats.json').write_text(json.dumps({'counts':dict(stats),'sample_posts':a.sample_posts,'seed':a.seed,'relation_policy':'unknown when source has no relation metadata','source_read_only':True},ensure_ascii=False,indent=2),encoding='utf-8')
 print(json.dumps(dict(stats),ensure_ascii=False)); print('output:',a.output_dir)
if __name__=='__main__': main()
