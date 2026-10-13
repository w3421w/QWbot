import argparse,csv,json
from pathlib import Path
from common.core import ROOT
PROJECT=ROOT.parents[2]

def count_lines(p):
 with p.open(encoding='utf-8-sig',newline='') as f: return sum(1 for _ in csv.reader(f)) if p.suffix == '.csv' else sum(1 for _ in f)

def entry(name,path,kind,count=None,files=1,fmt=None,platform='unknown',note='',speaker=None,parent=None,time=None,real=True,license='verification_pending'):
 return dict(source_name=name,source_path=str(path),platform=platform,dataset_type=kind,record_count=count,file_count=files,format=fmt or Path(path).suffix.lstrip('.'),license_status=license,provenance=note,has_speaker_id=speaker,has_parent_id=parent,has_timestamp=time,has_real_text=real,review_status='unreviewed',trainable=False)

def inventory():
 d=PROJECT/'datasets'; e=PROJECT/'experiments/qwen_fourway'; out=[]
 p=d/'zh_meme_sft_8k_review_candidates_20261009.csv'
 if p.exists(): out.append(entry('zh_meme_sft_8k review candidates',p,'C_review_only',count_lines(p)-1,platform='unknown',note='SFT candidate pairs; not Bilibili comments; origin rights unverified',speaker=False,parent=False,time=False))
 raw=d/'zh_meme_sft_8k_raw_20261009'
 for p in sorted(raw.glob('*.jsonl')): out.append(entry('zh_meme_sft_8k '+p.stem,p,'A_raw_dataset',count_lines(p),note='Imported external dataset; not verified community comment export',speaker=False,parent=False,time=False))
 raw_k=d/'kangba_raw/data'
 if raw_k.exists(): out.append(entry('Kangba raw Tieba post pages',raw_k,'A_raw_data',None,files=sum(1 for _ in raw_k.rglob('*.json')),fmt='json',platform='tieba',note='Raw post pages; record count not measured; reuse rights pending'))
 p=e/'corpus_pipeline/v2_1/output_full_repaired_01/style_candidates.jsonl'
 if p.exists(): out.append(entry('Kangba V2.1 style candidates',p,'B_cleaned',count_lines(p),platform='tieba',note='Existing derived style candidates; source/license audit separate'))
 for name,p in [('SmoothConv education',e/'conversation_pipeline/output_smoothconv_pilot_v2/messages.jsonl'),('SmoothConv social',e/'conversation_pipeline/output_smoothconv_social_audit_v2/messages.jsonl'),('MPDD',e/'conversation_pipeline/output_mpdd_pilot/messages.jsonl')]:
  if p.exists(): out.append(entry(name,p,'B_cleaned',count_lines(p),platform='other',note='Existing conversation pipeline output; not community platform comments'))
 p=e/'data_audit/new_corpus_collection_plan.md'; out.append(entry('new corpus collection plan',p,'D_plan_only',None,platform='unknown',note='Targets are proposed, not downloaded',real=False))
 p=e/'conversation_pipeline/07_fetch_irc.py'; out.append(entry('IRC importer',p,'E_code_unverified',None,platform='irc',note='Code and synthetic tests; no real import',real=False))
 for platform in ('bilibili','xiaohongshu'):
  out.append(entry(platform+' offline community export',ROOT/'output_community_pilot'/platform,'D_not_acquired',None,files=0,platform=platform,note='No verified local platform comment export identified',real=None))
 return out

def main():
 p=argparse.ArgumentParser(); p.add_argument('--output-json',default=str(ROOT/'reports/existing_corpus_inventory.json')); p.add_argument('--output-md',default=str(ROOT/'reports/existing_corpus_inventory.md')); a=p.parse_args()
 rows=inventory(); Path(a.output_json).write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n',encoding='utf-8') if not Path(a.output_json).exists() else (_ for _ in ()).throw(FileExistsError(a.output_json))
 lines=['# 现有语料盘点','','只读元数据盘点；未知数量使用 null。所有来源 trainable=false。','', '|来源|类别|记录数|许可|说明|','|---|---|---:|---|---|']
 for r in rows: lines.append(f"|{r['source_name']}|{r['dataset_type']}|{r['record_count'] if r['record_count'] is not None else 'unknown'}|{r['license_status']}|{r['provenance']}|")
 with open(a.output_md,'x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
 print(len(rows),'sources inventoried')
if __name__=='__main__': main()
