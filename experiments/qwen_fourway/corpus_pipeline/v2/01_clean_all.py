#!/usr/bin/env python3
"""Read-only source scan; deterministic two-pass V2 candidate construction."""
from __future__ import annotations
import argparse, collections, csv, hashlib, importlib.util, json, random, re, shutil, stat, sys
from difflib import SequenceMatcher
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
RAW = ROOT / 'datasets/kangba_raw/data'
TEST = ROOT / 'datasets/QWbot_V44_真实群聊影子测试_匿名31题.jsonl'
spec = importlib.util.spec_from_file_location('quality', HERE / '02_quality_grade.py')
quality = importlib.util.module_from_spec(spec); spec.loader.exec_module(quality)

def digest(text): return hashlib.sha256(text.encode('utf-8')).hexdigest()
def compact(text): return re.sub(r'\W+', '', quality.norm(text).casefold())
def post_id(url, path, index):
    match = re.search(r'/p/(\d+)', str(url))
    return match.group(1) if match else 'post-' + digest(f'{path}:{index}:{url}')[:16]

def simhash(text):
    chars = compact(text)
    grams = set(chars[i:i+3] for i in range(len(chars)-2))
    weights = [0] * 64
    for gram in grams:
        h = int.from_bytes(hashlib.blake2b(gram.encode(), digest_size=8).digest(), 'big')
        for bit in range(64): weights[bit] += 1 if h & (1 << bit) else -1
    return sum(1 << bit for bit, value in enumerate(weights) if value > 0)

def near(a, b):
    if min(len(a), len(b)) < 8 or max(len(a), len(b)) > min(len(a), len(b)) * 1.18: return False
    return SequenceMatcher(None, a, b, autojunk=False).ratio() >= .90

class NearIndex:
    """Four 16-bit SimHash bands: <=3-bit Hamming neighbors share a band."""
    def __init__(self): self.bands = collections.defaultdict(set); self.items = []
    def add(self, text, record_id):
        key = compact(text)
        if len(key) < 8: return None
        h = simhash(key)
        ids = set()
        for band in range(4): ids.update(self.bands[(band, (h >> (16*band)) & 65535)])
        match = None
        for i in sorted(ids):
            old_h, old_key, old_id = self.items[i]
            if (h ^ old_h).bit_count() <= 3 and near(key, old_key): match = old_id; break
        i = len(self.items); self.items.append((h, key, record_id))
        for band in range(4): self.bands[(band, (h >> (16*band)) & 65535)].add(i)
        return match

class UnionFind:
    def __init__(self): self.parent = {}
    def root(self, x):
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]; x = self.parent[x]
        return x
    def union(self, a, b):
        a, b = self.root(a), self.root(b)
        if a != b: self.parent[max(a,b)] = min(a,b)

def test_index():
    if not TEST.is_file(): raise RuntimeError(f'missing frozen test set: {TEST}')
    rows = []
    with TEST.open(encoding='utf-8-sig') as f:
        for line in f:
            if line.strip(): rows.append(json.loads(line))
    if len(rows) != 31: raise RuntimeError(f'expected 31 test rows, found {len(rows)}')
    texts = []
    for row in rows:
        for name in ('incoming_message', 'example_reply_for_human_review'):
            if row.get(name): texts.append((row['id'], name, quality.norm(row[name])))
        for i, part in enumerate(row.get('context', [])):
            if isinstance(part, dict) and part.get('text'): texts.append((row['id'], f'context_{i}', quality.norm(part['text'])))
    exact = collections.defaultdict(list); grams = collections.defaultdict(set)
    for i, (tid, field, value) in enumerate(texts):
        exact[digest(value.casefold())].append(i)
        c = compact(value)
        if len(c) >= 8:
            for j in range(len(c)-2): grams[c[j:j+3]].add(i)
    return rows, texts, exact, grams

def test_hit(value, test_data):
    _, texts, exact, grams = test_data
    value = quality.norm(value)
    if not value: return None
    hits = exact.get(digest(value.casefold()))
    if hits:
        i = hits[0]; return {'test_id': texts[i][0], 'test_field': texts[i][1], 'match_type': 'exact'}
    key = compact(value)
    if len(key) < 8: return None
    counts = collections.Counter(i for j in range(len(key)-2) for i in grams.get(key[j:j+3], ()))
    for i, count in counts.most_common():
        test_key = compact(texts[i][2]); needed = max(2, int(.5 * min(len(key)-2, len(test_key)-2)))
        if count >= needed and near(key, test_key):
            return {'test_id': texts[i][0], 'test_field': texts[i][1], 'match_type': 'near'}
    return None

def source_files():
    files = sorted(RAW.rglob('tieba_post_detail_page_*.json'))
    if len(files) != 105: raise RuntimeError(f'expected 105 source JSON files, found {len(files)}')
    return files

def iter_posts(files, sample, seed):
    if sample is None:
        for path in files:
            data = json.loads(path.read_text(encoding='utf-8-sig'))
            if not isinstance(data, list): raise RuntimeError(f'not a JSON list: {path}')
            for i, item in enumerate(data): yield path, i, item
    else:
        rng = random.Random(seed); reservoir = []; seen = 0
        for path in files:
            data = json.loads(path.read_text(encoding='utf-8-sig'))
            if not isinstance(data, list): raise RuntimeError(f'not a JSON list: {path}')
            for i, item in enumerate(data):
                seen += 1; rec = (path, i, item)
                if len(reservoir) < sample: reservoir.append(rec)
                else:
                    j = rng.randrange(seen)
                    if j < sample: reservoir[j] = rec
        yield from sorted(reservoir, key=lambda r:(str(r[0]),r[1]))

def write_jsonl(handle, row): handle.write(json.dumps(row, ensure_ascii=False, separators=(',',':')) + '\n')

def run(args):
    files = source_files(); tests = test_index()
    if args.check_only:
        posts = replies = 0
        for _, _, item in iter_posts(files, None, args.seed):
            if not isinstance(item, dict) or not isinstance(item.get('回复列表'), list): raise RuntimeError('invalid source post')
            posts += 1; replies += len(item['回复列表'])
        print(json.dumps({'source_files':len(files),'posts':posts,'replies':replies,'test_rows':len(tests[0]),'test_sha256':hashlib.sha256(TEST.read_bytes()).hexdigest(),'output_exists':args.output_dir.exists()},ensure_ascii=False)); return
    if args.output_dir.exists() or args.review_dir.exists(): raise RuntimeError('target output directory already exists; choose new version paths')
    if args.sample_posts is not None and args.sample_posts <= 0: raise RuntimeError('--sample-posts must be positive')
    args.output_dir.mkdir(parents=True, mode=0o700); args.review_dir.mkdir(parents=True, mode=0o700)
    tmp = args.output_dir / '_stage.jsonl'
    counts = collections.Counter(); grades = collections.Counter(); reasons_count = collections.Counter()
    dup = collections.defaultdict(list); first = {}; near_idx = NearIndex(); near_edges = []; uf = UnionFind(); overlaps = []
    post_test = {}; source_pids = collections.Counter(); samples = collections.defaultdict(list)
    with tmp.open('x', encoding='utf-8') as stage:
        for path, pi, item in iter_posts(files, args.sample_posts, args.seed):
            if not isinstance(item, dict) or not isinstance(item.get('回复列表'), list): raise RuntimeError(f'invalid post {path}:{pi}')
            counts['posts'] += 1
            relpath = str(path.relative_to(ROOT)); pid = post_id(item.get('链接'), relpath, pi)
            source_pids[pid] += 1; uf.root(pid)
            title_raw = item.get('标题') if isinstance(item.get('标题'), str) else ''
            body_raw = item.get('楼主内容') if isinstance(item.get('楼主内容'), str) else ''
            title = quality.norm(title_raw); body = quality.norm(body_raw)
            context_privacy = quality.privacy_kinds(title+' '+body)
            phit = test_hit(title, tests) or test_hit(body, tests)
            post_test[pid] = phit or post_test.get(pid)
            if phit: counts['test_overlap_posts'] += 1
            for ri, raw in enumerate(item['回复列表']):
                counts['replies'] += 1
                if not isinstance(raw, str): raise RuntimeError(f'non-string reply {path}:{pi}:{ri}')
                normalized = quality.norm(raw); h = digest(normalized.casefold()); rid = f'{pid}:{digest(relpath)[:8]}:{pi}:{ri}:{h[:12]}'
                reasons = []; flags = []
                bad = quality.spam_or_garbage(normalized)
                if bad: reasons.append(bad)
                privacy = sorted(set(quality.privacy_kinds(normalized) + context_privacy))
                if privacy: reasons.append('privacy_quarantine'); flags.extend('privacy_'+p for p in privacy)
                if quality.RUMOR.search(normalized): flags.append('unverified_personal_rumor')
                old = first.get(h)
                if old:
                    reasons.append('exact_duplicate'); flags.append('cross_post_duplicate' if old[1] != pid else 'within_post_duplicate'); uf.union(pid, old[1]); counts['exact_duplicate_occurrences'] += 1
                else: first[h] = (rid, pid)
                dup[h].append((rid, pid))
                near_of = None
                if not old and len(compact(normalized)) >= 8:
                    near_of = near_idx.add(normalized, rid)
                    if near_of:
                        flags.append('near_duplicate_review'); counts['near_duplicate_occurrences'] += 1
                        other_pid = near_of.split(':',1)[0]; uf.union(pid, other_pid)
                        near_edges.append((near_of,rid))
                rhit = test_hit(normalized, tests)
                hit = phit or rhit
                if hit:
                    post_test[pid] = hit
                    reasons.append('test_overlap_isolation'); counts['test_overlap_replies'] += 1
                    overlaps.append({'id':rid,'source_post_id':pid,'source_file':relpath,'reply_index':ri,'matched_field':'post_context' if phit else 'reply','test_id':hit['test_id'],'test_field':hit['test_field'],'match_type':hit['match_type'],'text_hash':h})
                if privacy: counts['privacy_risk'] += 1
                if bad == 'advertisement': counts['advertisement'] += 1
                if bad in ('garbled','machine_repetition','empty','oversize_spam_review'): counts['garbage'] += 1
                for reason in reasons: reasons_count[reason] += 1
                rumor = 'unverified_personal_rumor' in flags
                grade, components = quality.grade(normalized,title,body,reasons,bool(near_of),rumor)
                evidence = quality.semantic_evidence(title,body,normalized)
                sft = bool(evidence['possible'] and grade in ('A','B') and not hit and not near_of)
                priority = 'high' if sft and evidence['score'] >= 4 and body else 'normal'
                row = {'id':rid,'source':'tieba_kangba_raw','source_post_id':pid,'source_post_index':pi,'source_file':relpath,'source_url':item.get('链接',''),'reply_index':ri,
                       'post_title_original':title_raw,'op_body_original':body_raw,'original_reply':raw,'post_title':quality.redact(title),'op_body':quality.redact(body),'normalized_reply':normalized,'candidate_text':quality.redact(normalized),
                       'text_hash':h,'raw_text_hash':digest(raw),'normalization':'NFKC_whitespace_v2','reply_relation':'unknown','relation_evidence':[],'author':None,'timestamp':None,'reply_target':None,
                       'license_status':'unconfirmed_conflicting_source_metadata','review_status':'pending_human_review' if grade != 'REJECT' else 'auto_quarantined','quality_grade':grade,'quality_components':components,
                       'quality_flags':flags,'reject_reasons':reasons,'exact_duplicate_of':old[0] if old else None,'near_duplicate_of':near_of,'test_overlap':hit,'semantic_evidence':evidence,
                       'sft_candidate':sft,'sft_review_priority':priority if sft else None,'provenance_type':'real_reply','trainable':False}
                write_jsonl(stage,row)
    # Hold out entire duplicate-connected source components if any member overlaps the independent test.
    blocked_roots = {uf.root(pid) for pid, hit in post_test.items() if hit}
    with tmp.open(encoding='utf-8') as stage, \
         (args.output_dir/'style_candidates.jsonl').open('x',encoding='utf-8') as style, \
         (args.output_dir/'sft_candidates.jsonl').open('x',encoding='utf-8') as sft, \
         (args.output_dir/'review_queue.jsonl').open('x',encoding='utf-8') as review, \
         (args.output_dir/'rejected.jsonl').open('x',encoding='utf-8') as rejected, \
         (args.output_dir/'test_overlap_review.jsonl').open('x',encoding='utf-8') as test_out:
        for line in stage:
            row = json.loads(line); root = uf.root(row['source_post_id'])
            row['source_group_id'] = root
            row['data_split'] = 'test_quarantine' if root in blocked_roots else ('validation' if int(digest(root)[:8],16) % 10 == 0 else 'train')
            if root in blocked_roots and 'test_overlap_isolation' not in row['reject_reasons']:
                row['reject_reasons'].append('test_group_isolation'); row['quality_grade']='REJECT'; row['review_status']='auto_quarantined'; row['sft_candidate']=False
                reasons_count['test_group_isolation'] += 1
            grades[row['quality_grade']] += 1
            if row['data_split'] == 'test_quarantine':
                counts['test_quarantined_replies'] += 1
                if not row['test_overlap']:
                    write_jsonl(test_out, {'id':row['id'],'source_post_id':row['source_post_id'],'source_group_id':root,'match_type':'source_group','text_hash':row['text_hash']})
            if row['quality_grade'] == 'REJECT':
                write_jsonl(rejected,row); counts['rejected'] += 1
            else:
                write_jsonl(style,row); counts['style_candidates'] += 1
                if row['sft_candidate']: write_jsonl(sft,row); counts['sft_candidates'] += 1
                write_jsonl(review,row); counts['review_queue'] += 1
            if len(samples[row['quality_grade']]) < 3:
                samples[row['quality_grade']].append({'id':row['id'],'text':row['candidate_text'][:90],'reasons':row['reject_reasons'],'flags':row['quality_flags']})
        for hit in overlaps: write_jsonl(test_out,hit)
    tmp.unlink()
    # Exact duplicate groups are complete, including rejected occurrences. Near edges are review evidence.
    with (args.output_dir/'duplicate_groups.jsonl').open('x',encoding='utf-8') as out:
        for h, members in sorted(dup.items()):
            if len(members)>1: write_jsonl(out, {'kind':'exact','text_hash':h,'occurrences':len(members),'source_post_count':len({p for _,p in members}),'member_ids':[rid for rid,_ in members]}); counts['exact_duplicate_groups'] += 1
        for a,b in near_edges:
            write_jsonl(out, {'kind':'near','member_ids':[a,b],'method':'simhash_64_4x16_hamming_le_3_plus_sequence_ratio_ge_0.90'}); counts['near_duplicate_edges'] += 1
    counts['source_files'] = len(files); counts['unique_post_ids'] = len(source_pids)
    if args.sample_posts is None and (counts['posts'],counts['replies']) != (5136,118446): raise RuntimeError('full corpus source count differs from audited baseline')
    if counts['style_candidates']+counts['rejected'] != counts['replies']: raise RuntimeError('reply conservation failed')
    stats = {'counts':dict(counts),'grade_distribution':dict(grades),'reason_occurrences':dict(reasons_count),'test_rows':len(tests[0]),'test_sha256':hashlib.sha256(TEST.read_bytes()).hexdigest(),
             'seed':args.seed,'sample_posts':args.sample_posts,'source_read_only':True,'license_status':'unconfirmed_conflicting_source_metadata',
             'license_note':'raw README front matter says MIT; citation section says CC BY-NC 4.0; no authorization inferred',
             'relation_policy':'all reply relations unknown; semantic candidate does not verify relation',
             'relation_distribution':{'unknown':counts['replies']},
             'review_status_distribution':{'pending_human_review':counts['style_candidates'],'auto_quarantined':counts['rejected']},
             'license_distribution':{'unconfirmed_conflicting_source_metadata':counts['replies']},
             'split_policy':'SHA256 of union-find duplicate-connected post group, 90/10; test-overlap groups quarantined',
             'formal_trainable_count':0,'sample_examples':dict(samples)}
    (args.output_dir/'quality_statistics.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2),encoding='utf-8')
    report = f"# V2 全量语料清洗报告\n\n帖子 {counts['posts']}；回复 {counts['replies']}；风格候选 {counts['style_candidates']}；SFT 待审核候选 {counts['sft_candidates']}；隔离/拒绝 {counts['rejected']}。\n\n所有回复关系均为 unknown；许可未核实，人工审核未完成；正式可训练 messages 数量为 0。\n\n质量等级：{dict(grades)}。\n\n重复：完全重复出现 {counts['exact_duplicate_occurrences']}，近似重复出现 {counts['near_duplicate_occurrences']}。广告 {counts['advertisement']}，垃圾 {counts['garbage']}，隐私风险 {counts['privacy_risk']}。测试字面命中回复 {counts['test_overlap_replies']}，整组隔离回复 {counts['test_quarantined_replies']}。\n\n规则、边界与复核命令见 README.md。自动评级仅用于分层审核。\n"
    (args.output_dir/'quality_report.md').write_text(report,encoding='utf-8')
    for p in args.output_dir.iterdir(): p.chmod(0o600)
    print(json.dumps(stats['counts'],ensure_ascii=False)); print('output:',args.output_dir)

def main():
    p=argparse.ArgumentParser(description='V2 read-only full Kangba cleaner; never writes training messages.')
    p.add_argument('--check-only',action='store_true',help='validate all sources and the frozen test set without writing')
    p.add_argument('--sample-posts',type=int,default=None,help='fixed-seed reservoir post sample')
    p.add_argument('--seed',type=int,default=20261009)
    p.add_argument('--output-dir',type=Path,default=HERE/'output_full')
    p.add_argument('--review-dir',type=Path,default=HERE/'output_review')
    run(p.parse_args())
if __name__=='__main__': main()
