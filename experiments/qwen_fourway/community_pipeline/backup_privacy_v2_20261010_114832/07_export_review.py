import argparse,csv,json,re
from pathlib import Path
from common.core import read_records,privacy

FIELDS=['message_id','platform','source_file','source_record_id','content_type','thread_id','parent_comment_id','root_comment_id','speaker_id','message_order','created_at','text','domain','topics','subtopics','classification_confidence','classification_evidence','quality_flags','privacy_flags','license_status','review_status','trainable']
def export(rows,path):
 path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('x',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=FIELDS); w.writeheader()
  for r in rows:
   safe,flags=privacy(r.get('text',''))
   old_flags=r.get('privacy_flags') if isinstance(r.get('privacy_flags'),list) else []
   if 'private_identity' in flags or 'private_identity' in old_flags: continue
   d={k:r.get(k) for k in FIELDS}; d['text']=safe
   d['source_file']=Path(r.get('source_file','')).name
   d['trainable']=False
   all_flags=set(flags) | set(old_flags)
   # Re-scan every scalar that is written to the audit file.  This keeps a
   # nickname embedded in a source filename or timestamp from bypassing the
   # same privacy rule used for message text.
   for k,v in list(d.items()):
    if isinstance(v,str):
     d[k],extra_flags=privacy(v)
     all_flags.update(extra_flags)
   d['privacy_flags']=sorted(all_flags)
   for k in ('domain','topics','subtopics','classification_evidence','quality_flags','privacy_flags'):
    value=d[k] if d[k] is not None else []
    d[k]=json.dumps(value,ensure_ascii=False)
   for k,v in d.items():
    if isinstance(v,str) and v[:1] in ('=','+','-','@'): d[k]="'"+v
   w.writerow(d)

def main():
 p=argparse.ArgumentParser(); p.add_argument('--input',required=True); p.add_argument('--output',required=True); a=p.parse_args()
 export([r for _,r in read_records(a.input)],a.output)
if __name__=='__main__': main()
