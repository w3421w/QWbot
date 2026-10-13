import argparse
from difflib import SequenceMatcher
from common.core import normalized, read_records, write_new

def inspect(rows, blind_texts=()):
 results=[]; by_id={}; by_thread_text={}; by_text={}
 for r in rows:
  mid=r['message_id']; key=normalized(r.get('text','')); thread=r.get('thread_id'); platform=r.get('platform')
  if mid in by_id: results.append({'type':'same_id','source_ids':[by_id[mid],mid],'basis':'identical namespaced ID','disposition':'review'})
  if key and (thread,key) in by_thread_text: results.append({'type':'same_thread_exact','source_ids':[by_thread_text[(thread,key)],mid],'basis':'NFKC whitespace-insensitive exact','disposition':'review'})
  if key and key in by_text and by_text[key][1]!=thread:
   other=by_text[key]
   results.append({'type':'cross_platform_exact' if other[2]!=platform else 'cross_thread_exact','source_ids':[other[0],mid],'basis':'NFKC whitespace-insensitive exact','disposition':'review'})
  by_id.setdefault(mid,mid); by_thread_text.setdefault((thread,key),mid); by_text.setdefault(key,(mid,thread,platform))
 keys=[(r['message_id'],normalized(r.get('text','')),r.get('thread_id'),r.get('platform')) for r in rows]
 for i,a in enumerate(keys):
  for b in keys[i+1:]:
   if a[2]==b[2] or not a[1] or a[1]==b[1] or abs(len(a[1])-len(b[1]))>max(3,int(.15*max(len(a[1]),len(b[1])))): continue
   if len(a[1])<12 or len(b[1])<12: continue
   score=SequenceMatcher(None,a[1],b[1]).ratio()
   if score>=.88: results.append({'type':'cross_platform_near' if a[3]!=b[3] else 'cross_thread_near','source_ids':[a[0],b[0]],'basis':f'SequenceMatcher={score:.3f}','disposition':'manual_review_only'})
 blind=[normalized(x) for x in blind_texts if isinstance(x,str)]
 overlap={'exact':0,'near':0,'status':'measured' if blind_texts else 'not_measured'}
 if blind:
  for _,key,_,_ in keys:
   if not key: continue
   if key in blind: overlap['exact']+=1
   elif len(key)>=12 and any(abs(len(key)-len(x))<=max(3,int(.15*max(len(key),len(x)))) and SequenceMatcher(None,key,x).ratio()>=.88 for x in blind if len(x)>=12): overlap['near']+=1
 return results,overlap

def main():
 p=argparse.ArgumentParser(description='Duplicate candidates and blind overlap counts; never deletes records')
 p.add_argument('--input',required=True); p.add_argument('--output',required=True); p.add_argument('--blind-test')
 a=p.parse_args(); rows=[r for _,r in read_records(a.input)]; blind=[]
 if a.blind_test:
  for _,r in read_records(a.blind_test):
   if isinstance(r,dict):
    def strings(x):
     if isinstance(x,str): yield x
     elif isinstance(x,list):
      for y in x: yield from strings(y)
     elif isinstance(x,dict):
      for k,y in x.items():
       if k not in ('answer','response','assistant','label'): yield from strings(y)
    blind.extend(strings(r))
 results,overlap=inspect(rows,blind)
 write_new(a.output,{'duplicate_candidates':results,'blind_overlap_counts':overlap,'records_examined':len(rows)})
 print(len(results),'duplicate candidates')
if __name__=='__main__': main()
