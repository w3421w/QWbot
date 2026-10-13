#!/usr/bin/env python3
import csv,json,os,re
from datetime import datetime,timezone
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request,ProxyHandler,build_opener
OUT=Path('datasets/raw_external'); REPORT=Path('external_download_report.json')
HF_ENDPOINT=os.environ.get('HF_ENDPOINT','https://hf-mirror.com').rstrip('/')
SOURCES=[('LCCC','thu-coai/LCCC-base'),('Firefly中文对话','YeungNLP/firefly-train-1.1M'),('BelleGroup_train_0.5M_CN','BelleGroup/train_0.5M_CN')]
DATA_RE=re.compile(r'\.(jsonl?|ndjson|csv|parquet|txt)(\.gz)?$',re.I)
def opener_from_env():
 p={}
 for s in ('http','https'):
  v=os.environ.get(s.upper()+'_PROXY') or os.environ.get(s.lower()+'_proxy')
  if v:p[s]=v
 return build_opener(ProxyHandler(p)),p
def fetch(opener,url,timeout=45):
 req=Request(url,headers={'User-Agent':'external-dataset-downloader/2.0','Accept':'*/*'})
 with opener.open(req,timeout=timeout) as r:return r.read()
def count_records(path):
 try:
  s=path.suffix.lower()
  if s in ('.jsonl','.ndjson'):return sum(bool(x.strip()) for x in path.open(encoding='utf-8',errors='replace'))
  if s=='.json':
   x=json.loads(path.read_text(encoding='utf-8'));return len(x) if isinstance(x,list) else 1
  if s=='.csv':
   with path.open(encoding='utf-8',errors='replace',newline='') as f:return max(0,sum(1 for _ in csv.reader(f))-1)
 except Exception:return None
 return None
def check(opener,url,timeout=20):
 try:fetch(opener,url,timeout);return {'url':url,'ok':True}
 except Exception as e:return {'url':url,'ok':False,'error':f'{type(e).__name__}: {e}'}
def main():
 OUT.mkdir(parents=True,exist_ok=True);opener,proxies=opener_from_env()
 diagnostics={'HF_ENDPOINT':HF_ENDPOINT,'network_status':check(opener,HF_ENDPOINT),'proxies':proxies,'dataset_repository_access':[]};results=[]
 for name,repo in SOURCES:
  api=f'{HF_ENDPOINT}/api/datasets/{quote(repo,safe="/")}/tree/main?recursive=true&expand=true';access=check(opener,api);access['repo']=repo;diagnostics['dataset_repository_access'].append(access)
  item={'dataset':name,'repo':repo,'file':None,'path':None,'size_bytes':None,'records':None,'status':'failed'}
  try:
   if not access['ok']:raise RuntimeError(access.get('error','repository API unavailable'))
   tree=json.loads(fetch(opener,api,60));c=[x for x in tree if x.get('type')=='file' and DATA_RE.search(x.get('path',''))]
   if not c:raise RuntimeError('repository contains no supported data file')
   c.sort(key=lambda x:(x.get('size') if isinstance(x.get('size'),int) else 10**30,len(x.get('path',''))));rel=c[0]['path'];target=OUT/re.sub(r'[^A-Za-z0-9_.-]+','_',name)/Path(rel).name;target.parent.mkdir(parents=True,exist_ok=True)
   payload=fetch(opener,f'{HF_ENDPOINT}/datasets/{quote(repo,safe="/")}/resolve/main/{quote(rel,safe="/")}',600)
   if not payload:raise RuntimeError('download returned an empty file')
   tmp=target.with_suffix(target.suffix+'.part');tmp.write_bytes(payload);tmp.replace(target);item.update(file=Path(rel).name,path=str(target),size_bytes=target.stat().st_size,records=count_records(target),status='downloaded')
  except Exception as e:item['error']=f'{type(e).__name__}: {e}'
  results.append(item)
 REPORT.write_text(json.dumps({'generated_at':datetime.now(timezone.utc).isoformat(),'diagnostics':diagnostics,'datasets':results},ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(REPORT)
 for x in results:print(json.dumps(x,ensure_ascii=False))
if __name__=='__main__':main()
