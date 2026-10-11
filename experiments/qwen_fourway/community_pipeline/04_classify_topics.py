import argparse
from common.core import read_records, write_new
from common.classifier import classify

def main():
 p=argparse.ArgumentParser(description='Offline candidate topic labels; manual overrides keyed by message_id')
 p.add_argument('--input',required=True); p.add_argument('--output',required=True); p.add_argument('--manual-overrides')
 a=p.parse_args(); overrides={}
 if a.manual_overrides:
  import json
  with open(a.manual_overrides,encoding='utf-8') as f: overrides=json.load(f)
 rows=[]
 for _,r in read_records(a.input):
  if not isinstance(r,dict) or 'text' not in r: raise ValueError('Input must contain normalized records')
  r.update(classify(r['text'],manual=overrides.get(r['message_id'])))
  r['trainable']=False; rows.append(r)
 write_new(a.output,rows,jsonl=True); print(len(rows),'classified candidates')
if __name__=='__main__': main()
