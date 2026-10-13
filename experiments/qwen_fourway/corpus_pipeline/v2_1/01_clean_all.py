#!/usr/bin/env python3
"""V2.1 structural repair: post provenance groups are never inferred from text."""
from __future__ import annotations
import argparse, collections, csv, hashlib, importlib.util, json, os, random, re
from difflib import SequenceMatcher
from pathlib import Path

HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[3]
RAW=ROOT/'datasets/kangba_raw/data'; TEST=ROOT/'datasets/QWbot_V44_真实群聊影子测试_匿名31题.jsonl'
spec=importlib.util.spec_from_file_location('quality',HERE/'02_quality_grade.py'); quality=importlib.util.module_from_spec(spec); spec.loader.exec_module(quality)

def digest(s): return hashlib.sha256(s.encode('utf-8')).hexdigest()
def compact(s): return re.sub(r'\W+','',quality.norm(s).casefold())
def post_id(item):
    """Return the stable real-post identity, independent of source JSON file."""
    m=re.search(r'/p/(\d+)',str(item.get('链接','')))
    if m: return f'p-{m.group(1)}'
    # Records without a real Tieba id use stable content provenance.  The
    # collection record id remains separate and is used for row uniqueness.
    title=quality.norm(item.get('标题','') if isinstance(item.get('标题',''),str) else '')
    body=quality.norm(item.get('楼主内容','') if isinstance(item.get('楼主内容',''),str) else '')
    return 'fallback-'+digest(title+'\x1f'+body)[:24]
def simhash(s):
    grams={s[i:i+3] for i in range(len(s)-2)}; w=[0]*64
    for g in grams:
        h=int.from_bytes(hashlib.blake2b(g.encode(),digest_size=8).digest(),'big')
        for b in range(64): w[b]+=1 if h&(1<<b) else -1
    return sum(1<<b for b,v in enumerate(w) if v>0)
def near(a,b):
    if min(len(a),len(b))<8 or max(len(a),len(b))>min(len(a),len(b))*1.18:return False
    return SequenceMatcher(None,a,b,autojunk=False).ratio()>=.90
class NearIndex:
    def __init__(self): self.bands=collections.defaultdict(set); self.grams=collections.defaultdict(set); self.items=[]
    def add_all(self,text,rid):
        key=compact(text)
        if len(key)<8:return []
        h=simhash(key); ids=set()
        keys={key[i:i+3] for i in range(len(key)-2)} if len(key)>=12 else set(key)
        for gram in keys: ids.update(self.grams[gram])
        for band in range(4): ids.update(self.bands[(band,(h>>(16*band))&65535)])
        matches=[]
        for i in sorted(ids):
            old_h,old_key,old_id=self.items[i]
            if near(key,old_key):
                matches.append(old_id)
        i=len(self.items); self.items.append((h,key,rid))
        for gram in keys: self.grams[gram].add(i)
        for band in range(4): self.bands[(band,(h>>(16*band))&65535)].add(i)
        return matches
    def add(self,text,rid):
        """Compatibility helper returning the first match, if any."""
        matches=self.add_all(text,rid)
        return matches[0] if matches else None
class UF:
    def __init__(self): self.p={}
    def root(self,x):
        self.p.setdefault(x,x)
        if self.p[x]!=x:self.p[x]=self.root(self.p[x])
        return self.p[x]
    def union(self,a,b):
        a,b=self.root(a),self.root(b)
        if a!=b:self.p[max(a,b)]=min(a,b)

def test_index():
    rows=[json.loads(x) for x in TEST.read_text(encoding='utf-8-sig').splitlines() if x.strip()]
    if len(rows)!=31: raise RuntimeError(f'expected 31 test rows, found {len(rows)}')
    vals=[]
    for r in rows:
        for k in ('incoming_message','example_reply_for_human_review'):
            if r.get(k): vals.append((r['id'],k,quality.norm(r[k])))
        for i,p in enumerate(r.get('context',[])):
            if isinstance(p,dict) and p.get('text'): vals.append((r['id'],f'context_{i}',quality.norm(p['text'])))
    exact=collections.defaultdict(list); grams=collections.defaultdict(set)
    for i,(_,_,v) in enumerate(vals):
        exact[digest(v.casefold())].append(i); c=compact(v)
        for j in range(max(0,len(c)-2)): grams[c[j:j+3]].add(i)
    return vals,exact,grams
def test_hit(value,data):
    vals,exact,grams=data; value=quality.norm(value)
    h=exact.get(digest(value.casefold()))
    if h:
        i=h[0]; return {'test_id':vals[i][0],'test_field':vals[i][1],'match_type':'exact'}
    c=compact(value)
    if len(c)<8:return None
    counts=collections.Counter(i for j in range(len(c)-2) for i in grams.get(c[j:j+3],()))
    for i,n in counts.most_common():
        tc=compact(vals[i][2]); needed=max(2,int(.5*min(len(c)-2,len(tc)-2)))
        if n>=needed and near(c,tc): return {'test_id':vals[i][0],'test_field':vals[i][1],'match_type':'near'}
    return None
def files():
    fs=sorted(RAW.rglob('tieba_post_detail_page_*.json'))
    if len(fs)!=105: raise RuntimeError(f'expected 105 source JSON files, found {len(fs)}')
    return fs
def posts(fs,sample,seed):
    if sample is None:
        for p in fs:
            data=json.loads(p.read_text(encoding='utf-8-sig'))
            for i,x in enumerate(data):yield p,i,x
    else:
        rng=random.Random(seed); res=[]; seen=0
        for p in fs:
            data=json.loads(p.read_text(encoding='utf-8-sig'))
            for i,x in enumerate(data):
                seen+=1; item=(p,i,x)
                if len(res)<sample:res.append(item)
                else:
                    j=rng.randrange(seen)
                    if j<sample:res[j]=item
        yield from sorted(res,key=lambda z:(str(z[0]),z[1]))
def write(f,row):f.write(json.dumps(row,ensure_ascii=False,separators=(',',':'))+'\n')
def sanitize(r):
    # Downstream artifacts deliberately contain no raw title/body/reply/url.
    keep=['id','source_record_id','source_post_id','source_post_index','source_file','reply_index','post_title','op_body','candidate_text','text_hash','raw_text_hash','normalization','reply_relation','relation_evidence','author','timestamp','reply_target','license_status','review_status','quality_grade','quality_components','quality_flags','reject_reasons','text_duplicate_group','near_duplicate_of','test_overlap','semantic_evidence','sft_candidate','sft_review_priority','provenance_type','trainable','source_group_id','duplicate_component_id','data_split']
    return {k:r.get(k) for k in keep}
def run(a):
    fs=files(); tests=test_index()
    if a.check_only:
        p=r=0
        for _,_,x in posts(fs,None,a.seed): p+=1; r+=len(x.get('回复列表',[]))
        print(json.dumps({'source_files':len(fs),'posts':p,'replies':r,'test_rows':31},ensure_ascii=False)); return
    if a.output_dir.exists() or a.review_dir.exists(): raise RuntimeError('target output directory already exists')
    a.output_dir.mkdir(parents=True,mode=0o700); a.review_dir.mkdir(parents=True,mode=0o700)
    stage=a.output_dir/'_stage.jsonl'; audit=a.output_dir/'internal_audit_restricted.jsonl'
    exact=collections.defaultdict(list); first={}; nearidx=NearIndex(); posts_seen=collections.Counter(); counts=collections.Counter(); grades=collections.Counter(); reasons=collections.Counter(); overlaps=[]; post_test={}; duplicate_links=[]
    with stage.open('x',encoding='utf-8') as out, audit.open('x',encoding='utf-8') as aud:
      for path,pi,item in posts(fs,a.sample_posts,a.seed):
        pid=post_id(item); source_record_id='record-'+digest(f'{path.relative_to(ROOT)}:{pi}')[:24]; posts_seen[pid]+=1; title_raw=item.get('标题','') if isinstance(item.get('标题',''),str) else ''; body_raw=item.get('楼主内容','') if isinstance(item.get('楼主内容',''),str) else ''
        title=quality.norm(title_raw); body=quality.norm(body_raw); phit=test_hit(title,tests) or test_hit(body,tests)
        if phit: post_test[pid]=phit
        for ri,raw in enumerate(item.get('回复列表',[])):
          counts['replies']+=1; norm=quality.norm(raw); h=digest(norm.casefold()); rid=f'{source_record_id}:{ri}:{h[:12]}'; reasons_list=[]; flags=[]; bad=quality.spam_or_garbage(norm)
          if bad: reasons_list.append(bad)
          privacy=sorted(set(quality.privacy_kinds(norm+' '+title+' '+body)))
          if privacy: reasons_list+=['privacy_quarantine']; flags += ['privacy_'+x for x in privacy]; counts['privacy_risk']+=1
          old=first.get(h); first.setdefault(h,(rid,pid)); exact[h].append((rid,pid))
          text_group='exact-'+h if old else None
          if old:
              reasons_list.append('exact_duplicate'); flags.append('cross_post_duplicate' if old[1]!=pid else 'within_post_duplicate'); counts['exact_duplicate_occurrences']+=1
              if old[1]!=pid: duplicate_links.append((rid,pid,old[0],old[1],'exact'))
          near_of=None
          near_matches=[]
          if not old and len(compact(norm))>=8:
            near_matches=nearidx.add_all(norm,rid)
            near_of=near_matches[0] if near_matches else None
            if near_matches:
              flags.append('near_duplicate_review'); counts['near_duplicate_occurrences']+=1
              for match in near_matches:
                other=match.split(':',1)[0]
                if other != pid: duplicate_links.append((rid,pid,match,other,'near'))
          if near_of and text_group is None: text_group='near-'+digest('|'.join(sorted([rid,near_of])))[:24]
          rhit=test_hit(norm,tests); hit=phit or rhit
          if hit:
            post_test[pid]=hit
            reasons_list.append('test_overlap_isolation'); overlaps.append({'id':rid,'source_post_id':pid,'test_id':hit['test_id'],'test_field':hit['test_field'],'match_type':hit['match_type'],'text_hash':h}); counts['test_overlap_replies']+=1
          grade,components=quality.grade(norm,title,body,reasons_list,bool(near_of),'unverified_personal_rumor' in flags); evidence=quality.semantic_evidence(title,body,norm); sft=bool(evidence['possible'] and grade in ('A','B') and not hit and not near_of)
          r={'id':rid,'source_record_id':source_record_id,'source_post_id':pid,'source_post_index':pi,'source_file':str(path.relative_to(ROOT)),'reply_index':ri,'post_title_original':title_raw,'op_body_original':body_raw,'original_reply':raw,'source_url':item.get('链接',''),'post_title':quality.redact(title),'op_body':quality.redact(body),'normalized_reply':norm,'candidate_text':quality.redact(norm),'text_hash':h,'raw_text_hash':digest(raw),'normalization':'NFKC_whitespace_v2','reply_relation':'unknown','relation_evidence':[],'author':None,'timestamp':None,'reply_target':None,'license_status':'unconfirmed_conflicting_source_metadata','review_status':'pending_human_review','quality_grade':grade,'quality_components':components,'quality_flags':flags,'reject_reasons':reasons_list,'text_duplicate_group':text_group,'near_duplicate_of':near_of,'test_overlap':hit,'semantic_evidence':evidence,'sft_candidate':sft,'sft_review_priority':'high' if sft and evidence['score']>=4 else ('normal' if sft else None),'provenance_type':'real_reply','trainable':False}
          write(out,r); write(aud,{'id':rid,'source_post_id':pid,'source_file':r['source_file'],'reply_index':ri,'post_title_original':title_raw,'op_body_original':body_raw,'original_reply':raw,'source_url':r['source_url']})
          counts['rejected' if grade=='REJECT' else 'style_candidates']+=1; grades[grade]+=1
          for q in reasons_list: reasons[q]+=1
    stage_rows=[json.loads(x) for x in stage.read_text(encoding='utf-8').splitlines() if x.strip()]
    by_id={r['id']:r for r in stage_rows}; post_split={pid:('validation' if int(digest(pid)[:8],16)%10==0 else 'train') for pid in posts_seen}
    for r in stage_rows:
      r['source_group_id']='post-'+r['source_post_id']; r['duplicate_component_id']='dupcomp-'+digest(r['id'])[:16]
      r['data_split']='test_quarantine' if r['source_post_id'] in post_test else post_split[r['source_post_id']]
    # Post assignment is primary. Only the concrete duplicate rows crossing the
    # boundary are quarantined; no text relation can merge source groups.
    duplicate_quarantine=set()
    for rid,pid,other_rid,other_pid,kind in duplicate_links:
      if post_split.get(pid) != post_split.get(other_pid):
        duplicate_quarantine.update((rid,other_rid)); counts[f'{kind}_cross_split_isolated']+=2
    for rid in duplicate_quarantine:
      if by_id[rid]['data_split'] != 'test_quarantine': by_id[rid]['data_split']='duplicate_quarantine'
      by_id[rid]['reject_reasons'].append('cross_split_duplicate_isolation')
    outputs={'style_candidates.jsonl':[],'sft_candidates.jsonl':[],'review_queue.jsonl':[],'rejected.jsonl':[],'test_quarantine.jsonl':[],'duplicate_quarantine.jsonl':[],'test_overlap_review.jsonl':overlaps}
    for r in stage_rows:
      s=sanitize(r)
      if r['data_split']=='test_quarantine': outputs['test_quarantine.jsonl'].append(s); continue
      if r['data_split']=='duplicate_quarantine': outputs['duplicate_quarantine.jsonl'].append(s); continue
      target='rejected.jsonl' if r['quality_grade']=='REJECT' else 'style_candidates.jsonl'; outputs[target].append(s)
      if target=='style_candidates.jsonl': outputs['review_queue.jsonl'].append(s); counts['review_queue']+=1
      if r['sft_candidate'] and target=='style_candidates.jsonl': outputs['sft_candidates.jsonl'].append(s); counts['sft_candidates']+=1
    stage.unlink()
    for name,rows in outputs.items():
      with (a.output_dir/name).open('x',encoding='utf-8') as f:
        for r in rows:write(f,r)
    with (a.output_dir/'duplicate_groups.jsonl').open('x',encoding='utf-8') as f:
      for h,m in sorted(exact.items()):
        if len(m)>1:write(f,{'kind':'exact','text_duplicate_group':'exact-'+h,'occurrences':len(m),'source_post_count':len({p for _,p in m}),'member_ids':[x for x,_ in m]})
    split_counts=collections.defaultdict(lambda: {'posts':0,'replies':0})
    for pid,sp in post_split.items(): split_counts['test_quarantine' if pid in post_test else sp]['posts']+=1
    for r in stage_rows: split_counts[r['data_split']]['replies']+=1
    final_counts={k:len(outputs[f'{k}.jsonl']) for k in ('style_candidates','sft_candidates','rejected','test_quarantine','duplicate_quarantine')}
    retained=[r for r in stage_rows if r['data_split'] in ('train','validation') and r['quality_grade']!='REJECT']
    final_counts.update({'train_replies':sum(r['data_split']=='train' and r['quality_grade']!='REJECT' for r in stage_rows),'validation_replies':sum(r['data_split']=='validation' and r['quality_grade']!='REJECT' for r in stage_rows),'train_posts':len({r['source_post_id'] for r in retained if r['data_split']=='train'}),'validation_posts':len({r['source_post_id'] for r in retained if r['data_split']=='validation'})})
    counts.update({'posts':sum(posts_seen.values()),'unique_posts':len(posts_seen),'source_files':len(fs),'test_quarantined_replies':final_counts['test_quarantine'],'duplicate_quarantined_replies':final_counts['duplicate_quarantine']})
    stats={'counts':dict(counts),'final_candidates':final_counts,'grade_distribution':dict(grades),'reason_occurrences':dict(reasons),'split_counts':dict(split_counts),'original_reply_total':counts['replies'],'conservation_check':{'original_replies':counts['replies'],'final_output_rows':sum(final_counts[k] for k in ('style_candidates','rejected','test_quarantine','duplicate_quarantine')),'passed':sum(final_counts[k] for k in ('style_candidates','rejected','test_quarantine','duplicate_quarantine'))==counts['replies']},'test_rows':31,'test_sha256':hashlib.sha256(TEST.read_bytes()).hexdigest(),'seed':a.seed,'sample_posts':a.sample_posts,'source_group_policy':'one real post id per source_group_id; collection record id is separate','text_duplicate_policy':'normalized exact and high-information near duplicates checked across retained splits','split_policy':'deterministic post split; concrete cross-split exact/near rows quarantined','relation_distribution':{'unknown':counts['replies']},'license_status':'unconfirmed_conflicting_source_metadata','formal_trainable_count':0,'downstream_redacted_fields':['post_title_original','op_body_original','original_reply','source_url']}
    (a.output_dir/'quality_statistics.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2),encoding='utf-8'); (a.output_dir/'quality_report.md').write_text('# V2.1 结构修复报告\n\n所有关系为 unknown，许可 unconfirmed，trainable 恒为 false；不导出 messages。原始字段仅在 `internal_audit_restricted.jsonl`。正则无法识别全部个人信息，隐私隔离需人工复核。\n\n'+json.dumps(stats,ensure_ascii=False,indent=2),encoding='utf-8')
    for p in a.output_dir.iterdir():p.chmod(0o600)
    print(json.dumps(stats['counts'],ensure_ascii=False))
def main():
 p=argparse.ArgumentParser(description='V2.1 structural repair cleaner; no training export')
 p.add_argument('--check-only',action='store_true'); p.add_argument('--sample-posts',type=int); p.add_argument('--seed',type=int,default=20261009); p.add_argument('--output-dir',type=Path,default=HERE/'output_full'); p.add_argument('--review-dir',type=Path,default=HERE/'output_review'); run(p.parse_args())
if __name__=='__main__':main()
