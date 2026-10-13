#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, random, collections, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from pipeline_common import RAW_DIR, norm

def main():
 p=argparse.ArgumentParser(description='Inspect raw Kangba JSON structure without modifying it.')
 p.add_argument('--check-only',action='store_true',help='scan all raw files and print structural report')
 p.add_argument('--sample-posts',type=int,default=100); p.add_argument('--seed',type=int,default=42); p.add_argument('--output-dir',type=Path,default=Path('experiments/qwen_fourway/corpus_pipeline/output'))
 a=p.parse_args(); files=sorted(RAW_DIR.rglob('tieba_post_detail_page_*.json')); stats=collections.Counter(); field_counts=collections.Counter(); reply_types=collections.Counter(); examples={}; reservoir=[]; rng=random.Random(a.seed); seen=0
 for path in files:
  try: data=json.loads(path.read_text(encoding='utf-8'))
  except Exception as e: stats['file_errors']+=1; continue
  stats['files']+=1
  if not isinstance(data,list): stats['invalid_top_level']+=1; continue
  for ix,item in enumerate(data):
   stats['posts']+=1
   if not isinstance(item,dict): stats['invalid_posts']+=1; continue
   for k,v in item.items(): field_counts[k]+=1; examples.setdefault(k,{'type':type(v).__name__,'example':str(v)[:180]})
   replies=item.get('回复列表'); reply_types[type(replies).__name__]+=1
   if isinstance(replies,list): stats['replies']+=len(replies); stats['reply_strings']+=sum(isinstance(x,str) for x in replies)
   else: stats['invalid_reply_lists']+=1
   seen+=1
   if len(reservoir)<a.sample_posts: reservoir.append({'file':str(path),'index':ix,'title':norm(item.get('标题')),'body':norm(item.get('楼主内容')),'reply_count':len(replies) if isinstance(replies,list) else 0})
   else:
    j=rng.randrange(seen)
    if j<a.sample_posts: reservoir[j]={'file':str(path),'index':ix,'title':norm(item.get('标题')),'body':norm(item.get('楼主内容')),'reply_count':len(replies) if isinstance(replies,list) else 0}
 report={'raw_dir':str(RAW_DIR),'files':sorted(str(x) for x in files),'counts':dict(stats),'field_counts':dict(field_counts),'reply_value_types':dict(reply_types),'field_examples':examples,'relation_fields_present': [k for k in field_counts if any(t in k.lower() for t in ('作者','用户','楼层','时间','引用','回复','id','楼')) and k not in ('回复列表','楼主内容')], 'notes':['回复列表元素均为纯字符串时，无法恢复作者、楼层、时间、引用或回复对象。','链接中的 /p/<id> 可恢复帖子 ID；回复 ID 不存在，V1 使用稳定的 source_reply_id 哈希。'],'sample_posts':reservoir}
 a.output_dir.mkdir(parents=True,exist_ok=True); (a.output_dir/'raw_structure_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8'); print(json.dumps(report['counts'],ensure_ascii=False)); print('report:',a.output_dir/'raw_structure_report.json')
if __name__=='__main__': main()
