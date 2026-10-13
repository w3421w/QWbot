#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""QWbot V4.1 抗吧语料质量审计与下一轮候选扩充。

依赖 Python 标准库；只读 V4 CSV 和可选原始 kangba_train.jsonl。
运行:
  python audit_kangba_v41.py audit
  python audit_kangba_v41.py expand --limit 800
脚本仅输出 V4.1 新文件；永不自动批准候选、永不覆盖 V4 数据或模型。
"""
from __future__ import annotations
import argparse
import csv
import difflib
import hashlib
import json
import random
import re
import statistics
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'datasets'
CANDIDATES = DATA / 'kangba_v4_kangba_candidates.csv'
SEEDS = DATA / 'kangba_v4_kangba_seed_examples.csv'
SOURCE = DATA / 'kangba_train.jsonl'
OLD_TEST = DATA / 'kangba_v4_kangba_holdout_prompts.txt'
AUDIT = DATA / 'kangba_v41_review_current.csv'
PRIORITY = DATA / 'kangba_v41_recheck_priority.csv'
EXPAND = DATA / 'kangba_v41_new_candidates.csv'
TESTS = DATA / 'kangba_v41_new_holdout.csv'
REPORT = DATA / 'kangba_v41_diagnostics.json'
TRAIN_JSON = DATA / 'kangba_v4_kangba_train.jsonl'
VALID_JSON = DATA / 'kangba_v4_kangba_valid.jsonl'

# 独立新测试题。原创测试材料，不来自原始贴吧评论，不得用于训练。
TEST_BANK = [
 ('贷款吹与反串','这个产品才放出一张海报，朋友已经宣布行业第一了'),
 ('贷款吹与反串','他只参加过一次试镜，粉丝已经安排三部电影男主'),
 ('贷款吹与反串','比赛还没抽签，有人已经给战队P上冠军奖杯'),
 ('贷款吹与反串','才拿下一小局，群里有人说剩下的比赛不用看了'),
 ('贷款吹与反串','主播开播十分钟，弹幕已经宣布年度最佳节目'),
 ('贷款吹与反串','他才报名驾校，已经研究买什么跑车了'),
 ('回旋镖与清算','早上还劝别人少跟网友吵架，晚上自己对线到四点'),
 ('回旋镖与清算','他上周说再也不看这队比赛，今天直播间第一排'),
 ('回旋镖与清算','之前嘲讽别人买新手机，转头自己就下单了'),
 ('回旋镖与清算','别人犯同样的错他骂半天，轮到自己就说可以理解'),
 ('回旋镖与清算','他曾说这游戏无聊，最近每天都在催队友上线'),
 ('回旋镖与清算','开赛前说输不了，赛后开始研究裁判和天气'),
 ('粉丝争论与拆台','有人说赢了是偶像带队，输了是队友拖累'),
 ('粉丝争论与拆台','他评价比赛只看自己喜欢的选手有没有镜头'),
 ('粉丝争论与拆台','有人拿十年前的奖杯证明今天每一局都不会输'),
 ('粉丝争论与拆台','明明同一个失误，他夸自己人勇敢骂别人送'),
 ('粉丝争论与拆台','他把所有不同意见都当成黑粉，这怎么聊'),
 ('粉丝争论与拆台','输的时候说是娱乐赛，赢的时候又说含金量爆炸'),
 ('抽象短梗','群里一个人刚开始分析，大伙就知道又要发长文'),
 ('抽象短梗','他一句话发出去，评论区突然变成两个阵营'),
 ('抽象短梗','这人每次说自己路过，结果每个帖子都有他'),
 ('抽象短梗','有个人立了一个flag，群友已经开始倒计时'),
 ('抽象短梗','刚有人说别急，结果他自己连发十条消息'),
 ('抽象短梗','他说纯路人，点开主页全是同一个选手'),
 ('日常水帖接话','群里突然有人发红包，潜水的人全出来了'),
 ('日常水帖接话','我一到周末就开始想着下周要上班'),
 ('日常水帖接话','我说马上到，结果现在还没出门'),
 ('日常水帖接话','外卖到了我才发现自己忘记点米饭'),
 ('日常水帖接话','我刚整理完桌面，五分钟后又乱了'),
 ('日常水帖接话','群里聊得热火朝天，我一发消息直接冷场'),
 ('事实与边界','朋友说这选手今晚突然换队了，有官宣吗'),
 ('事实与边界','听说某位主播刚被处罚了，是不是真的'),
 ('事实与边界','这个老梗是哪个人最早说的，能确定吗'),
 ('事实与边界','有人发了张截图说是官方公告，能信吗'),
 ('事实与边界','这场比赛到底什么时候开打，别给我猜'),
 ('事实与边界','网上说一家公司倒闭了，有可靠消息吗'),
 ('认真回应','朋友家刚出了事，我现在很难受，能不能别开玩笑'),
 ('认真回应','我爸在医院，我现在真没心情整活'),
 ('认真回应','最近准备考试时总心慌，能认真给我出个主意吗'),
 ('认真回应','我今天被人误会了，真的挺委屈的'),
 ('认真回应','今天忙了一整天，我只想找个地方安静会儿'),
 ('认真回应','我状态不太好，能不能正常聊两句'),
]

KB_TOPIC = re.compile(r'抗吧|抗抗|吧友|串子|反串|回旋镖|贷款|清算|抽象|闹麻|典|绷|嘴硬|开香槟|粉丝|双标')
ESPORT = re.compile(r'(?i)\b(?:lol|lpl|lck|t1|tes|blg|jdg|ig|rng|msi|faker|elk|jkl|uzi|bin|xun|geng|wbg|nip|al|edg|gala|zeus|chovy)\b|比赛|选手|战队|冠军|赛事|赛区|打野|上单|下路|中单|辅助|对线|沙皇|大龙|小龙|团战|电竞|职业|季后赛|bp|英雄联盟|涅槃|碧螺春|水鬼|蝌蚪|马头|鸟波|大飞|尺帝')
SENSITIVE_FACT = re.compile(r'官宣|转会|禁赛|处罚|假赛|涉嫌|爆料|被抓|劈腿|出轨|女朋友|男朋友|怀孕|确诊|内部消息|内幕|最新消息|昨天.*赢|今天.*赢')
TOPIC_QUEST = re.compile(r'为什么|到底|怎么|是否|是不是真|真的假的|多少|哪年|什么时间|哪场|\?|？')
GENERIC_ANSWER = re.compile(r'^(?:典|绷|闹麻|确实|好吧|哈哈+|有一说一|我觉得|不是哥们|别急|是的|不是|对|啊+|哦+)[，,。!！?？ ]*$',re.I)
CHARS = re.compile(r'[\u4e00-\u9fff]|[a-z0-9]+', re.I)


def normalize(v):
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', str(v or ''))).strip()


def key(v):
    return re.sub(r'[^\w\u4e00-\u9fff]', '', normalize(v)).casefold()


def read_csv(path):
    if not path.is_file():
        raise FileNotFoundError(f'找不到 {path}。请把已审核的V4 CSV放到QWbot/datasets后再运行。')
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def csv_write(path, rows, fields):
    if path.exists():
        raise FileExistsError(f'已有文件 {path}，为了避免覆盖审核结果请先自行备份或重命名。')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        for r in rows:
            safe = {}
            for k in fields:
                v = str(r.get(k, '') or '')
                if v.lstrip().startswith(('=', '+', '-', '@', '\t', '\r')):
                    v = "'" + v
                safe[k] = v
            writer.writerow(safe)
    print('生成:', path, '| 行数:',len(rows))


def overlap(p,c):
    """启发式共享汉字双字片段；不是语义相关性的可靠判定。"""
    def grams(s):
        s = key(s)
        return {s[i:i+2] for i in range(len(s)-1) if re.search('[\u4e00-\u9fff]', s[i:i+2])}
    pg,cg = grams(p),grams(c)
    banal = {'这个','那个','怎么','什么','不是','就是','还是','可以','你们','他们','我们','一样','自己','现在','已经','不是','没有','因为','所以','一个','知道','感觉','真的','还有','比赛','谁说'}
    pg-=banal; cg-=banal
    return len(pg&cg), '/'.join(sorted(pg&cg)[:4])


def flags_for(row):
    p,c=normalize(row.get('prompt','')),normalize(row.get('completion',''))
    flags=[]
    if len(c)<4: flags.append('very_short_reply_review')
    if len(c)>34: flags.append('long_reply_review')
    if len(p)>110: flags.append('long_post_context_review')
    n,ov=overlap(p,c)
    if n==0: flags.append('no_shared_topic_words_review')
    if c and p and key(p)==key(c): flags.append('parrot_input')
    if GENERIC_ANSWER.fullmatch(c): flags.append('formulaic_short_reply')
    if SENSITIVE_FACT.search(c): flags.append('potential_unverified_claim')
    if TOPIC_QUEST.search(p) and (SENSITIVE_FACT.search(p) or any(t in p for t in ['出自','来源','哪场比赛','真的吗'])):
        flags.append('fact_question_requires_reference')
    # 实际主题 vs 原game字段冲突：来源可能就是电竞讨论，GENERAL不等于非电竞。
    if row.get('game')=='GENERAL' and ESPORT.search(p): flags.append('general_label_but_esports_text')
    if len(p)>0 and len(c)>0 and len(c)>len(p)*2 and len(c)>28: flags.append('reply_much_longer_than_prompt')
    if 'authored_example' == row.get('origin'): flags.append('authored_not_forum')
    return flags,n,ov


def ratio(a,b):
    return difflib.SequenceMatcher(None,key(a),key(b)).ratio()


def near_holdouts(rows):
    # 在新测试题上做“字面近似”防泄漏；语义相近的例子仍需人工判断。
    tests=[]
    for category,prompt in TEST_BANK:
        best=max(((ratio(prompt, r.get('prompt','')), r.get('id',''), r.get('prompt','')) for r in rows),default=(0,'',''))
        tests.append({'category':category,'prompt':prompt,'near_train_similarity':f'{best[0]:.3f}',
                      'closest_training_id':best[1], 'closest_training_prompt':best[2],
                      'semantic_leakage_warning':'人工核对' if best[0]>=.60 else ''})
    return tests


def audit(args):
    candidate=read_csv(CANDIDATES)
    seeds=read_csv(SEEDS)
    selected=[dict(r) for r in candidate+seeds if str(r.get('approve','')).lower()=='y']
    nonselected=[dict(r) for r in candidate if str(r.get('approve','')).lower()!='y']
    flag_count=Counter(); classes=Counter(); game=Counter(); audit_rows=[]
    for r in selected:
        flags,n,ov=flags_for(r)
        flag_count.update(flags)
        classes[r.get('category','')]+=1
        game[r.get('game','')]+=1
        audit_rows.append({**r, 'v41_shared_bigrams':n, 'v41_topic_overlap_examples':ov,
                           'v41_flags':';'.join(flags), 'v41_reaudit':'',
                           'v41_reason':'', 'v41_reference_checked':''})
    # 复核队列：之前n只是本轮暂缓，不自动改approve。尽量每个帖子最多一条。
    nonselected.sort(key=lambda x:(-int(x.get('score','0') or 0),x.get('id','')))
    covered_posts={r.get('post_id') for r in selected if r.get('origin')=='forum_candidate'}
    used_p=set(); priority=[]
    for r in nonselected:
        if len(priority)>=args.priority_limit: break
        if r.get('post_id','') in covered_posts: continue
        if r.get('post_id','') in used_p: continue
        fl,n,ov=flags_for(r)
        if 'potential_unverified_claim' in fl or 'parrot_input' in fl: continue
        # 低文本匹配或长文仍可复审，但排序靠后。
        used_p.add(r.get('post_id',''))
        priority.append({**r,'v41_shared_bigrams':n,'v41_flags':';'.join(fl),
                         'v41_reaudit':'', 'v41_reason':'', 'v41_reference_checked':''})
    # 将无明显警告的排前，但明确不是人工审核/自动批准。
    priority.sort(key=lambda r:(len(r['v41_flags'].split(';')) if r['v41_flags'] else 0,
                                -int(r.get('score') or 0)))
    audit_fields=list(dict.fromkeys(list(candidate[0].keys())+['v41_shared_bigrams','v41_topic_overlap_examples',
            'v41_flags','v41_reaudit','v41_reason','v41_reference_checked']))
    priority_fields=list(dict.fromkeys(list(candidate[0].keys())+['v41_shared_bigrams','v41_flags',
            'v41_reaudit','v41_reason','v41_reference_checked']))
    csv_write(AUDIT,audit_rows,audit_fields)
    csv_write(PRIORITY,priority,priority_fields)
    tests=near_holdouts(selected)
    csv_write(TESTS,tests,list(tests[0]))
    old_leaks=[]
    if OLD_TEST.is_file():
        for line in OLD_TEST.read_text(encoding='utf-8').splitlines():
            if not line.strip(): continue
            match=max(((ratio(line,r['prompt']),r['id'],r['prompt']) for r in selected),default=(0,'',''))
            if match[0]>=.48:
                old_leaks.append({'holdout':line,'similarity':round(match[0],3), 'id':match[1], 'train_prompt':match[2]})
    trainvalid={}
    if TRAIN_JSON.exists() and VALID_JSON.exists():
        def load_json_prompts(path):
            with path.open(encoding='utf-8') as f:
                for line in f:
                    if not line.strip():continue
                    rec=json.loads(line)
                    msgs=rec.get('prompt',[])
                    yield '\n'.join(x.get('content','') for x in msgs if x.get('role')=='user')
        tr=list(load_json_prompts(TRAIN_JSON));va=list(load_json_prompts(VALID_JSON))
        exact=set(key(x) for x in tr)&set(key(x) for x in va)
        similar=[]
        for v in va:
            highest=max(((ratio(v,t),t) for t in tr),default=(0,''))
            if highest[0]>=.76:
                similar.append({'valid_prompt':v,'train_prompt':highest[1],
                                'lexical_similarity':round(highest[0],3)})
        trainvalid={'train_count':len(tr),'valid_count':len(va),
                    'exact_shared_prompt_count':len(exact),
                    'high_similarity_prompt_pairs_ge_0_76':similar[:25],
                    'high_similarity_count':len(similar)}
    report={'inputs':{'v4_candidate_rows':len(candidate),'approved_forum':sum(r['approve']=='y' for r in candidate),
                       'original_seed_examples':len(seeds),'selected_including_seed':len(selected)},
            'selected_category_counts':dict(classes),'game_label_counts':dict(game),
            'median_selected_reply_length':statistics.median(len(r['completion']) for r in selected) if selected else 0,
            'audit_flags':dict(flag_count),'recheck_priority_count':len(priority),
            'new_holdout_count':len(tests), 'new_holdout_high_lexical_similarity':sum(float(t['near_train_similarity'])>=.60 for t in tests),
            'old_test_near_examples_above_0_48':old_leaks,
            'train_validation_near_duplicate_diagnostics':trainvalid,
            'important':'词面相似/关键词覆盖不是语义正确率；标记仅用于复核。评估题不纳入训练。以前approve=n的样本不会自动转成y。'
    }
    if REPORT.exists(): raise FileExistsError(f'报告已存在 {REPORT}。请先备份或重命名。')
    REPORT.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('\nV4.1诊断结果：')
    print('已批准候选(含原创):',len(selected),' | 待复审池:',len(nonselected),' | 复审优先队列:',len(priority))
    print('机器复核警示:',dict(flag_count))
    print('旧测试集字面近似提醒数:',len(old_leaks),' | 新测试题:',len(tests))
    print('完整报告:',REPORT)


def expand(args):
    """从旧完整训练JSONL中抽取未出现在既有1,000行候选的新样本，仅输出待复审，不自动训练。"""
    reviewed=read_csv(CANDIDATES)
    if not SOURCE.is_file(): raise FileNotFoundError(f'找不到 {SOURCE}：新扩充必须在QWbot原数据环境下进行。')
    prev_pair={(key(r['prompt']),key(r['completion'])) for r in reviewed}
    prev_prompt={key(r['prompt']) for r in reviewed}
    used_pairs=set(prev_pair); by_cat=defaultdict(list); all_counts=Counter()
    # 导入既有筛选模块，保持相同的风险过滤规则，但不再强制拒绝词面弱匹配。
    import importlib.util
    code=ROOT/'prepare_kangba_v4_kangba.py'
    if not code.exists(): raise FileNotFoundError(f'缺少原脚本 {code}')
    spec=importlib.util.spec_from_file_location('v4_kangba_prepare',str(code)); mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    with SOURCE.open('r',encoding='utf-8') as handle:
        for line_num,line in enumerate(handle,1):
            if not line.strip(): continue
            all_counts['read']+=1
            try:
                item=json.loads(line)
                prompt='\n'.join(v.get('content','') for v in item['prompt'] if v.get('role')=='user')
                completion=item['completion'][-1]['content']
            except (KeyError,TypeError,ValueError,IndexError):
                all_counts['broken_row']+=1
                continue
            prompt,completion=normalize(prompt),normalize(completion)
            pkey,ckey=key(prompt),key(completion)
            if not pkey or not ckey or (pkey,ckey) in used_pairs:
                all_counts['duplicate_or_empty']+=1; continue
            if pkey in prev_prompt:
                all_counts['old_prompt_covered']+=1; continue
            if not (8<=len(prompt)<=180 and 3<=len(completion)<=42):
                all_counts['length_excluded']+=1; continue
            if mod.check_risk(prompt,completion):
                all_counts['risk_filtered']+=1; continue
            n,_=overlap(prompt,completion)
            game=mod.game_of(prompt)
            cat=mod.classify(prompt,completion,game)
            # 原算法优先高频抗吧词，这里弱化：话题相近性+简洁+回复非套话。
            score= min(n,5)*2 + (5 if len(completion)<=22 else 3 if len(completion)<=32 else 1)
            if mod.STYLE.search(completion): score+=1
            if GENERIC_ANSWER.fullmatch(completion): score-=4
            if SENSITIVE_FACT.search(completion):score-=6
            if score<args.min_score:
                all_counts['below_threshold']+=1;continue
            pid='prompt_'+hashlib.sha1(pkey.encode('utf-8')).hexdigest()[:18]
            pair_id='forum_'+hashlib.sha1((pkey+'\n'+ckey).encode('utf-8')).hexdigest()[:20]
            row={'approve':'','id':pair_id,'score':score,'category':cat,'game':game,
                 'origin':'forum_candidate','post_id':pid,'source_url':'',
                 'context_signals':f'v41_shared_bigrams={n}',
                 'review_warning':'未审核新样本；原始数据可能缺少真实楼中楼上下文；不可直接训练',
                 'prompt':prompt,'completion':completion,
                 'v41_review':'','v41_reviewer_note':'','v41_reference_checked':''}
            by_cat[cat].append(row); used_pairs.add((pkey,ckey))
            all_counts['eligible_new']+=1
    # 均衡话题；不假装有超过实际量的非电竞语料。
    quotas={'贷款吹与反串':.18,'回旋镖与清算':.16,'抽象短梗':.16,
            '粉丝争论与拆台':.16,'日常水帖接话':.19,'赛事梗与讨论':.10,
            '事实与边界':.03,'认真回应':.02}
    rng=random.Random(3407)
    for lst in by_cat.values():
        rng.shuffle(lst)
        lst.sort(key=lambda r:-r['score'])
    batch=[];used_prompt=set();used_reply=set();used_post=set()
    def add(row):
        pk,ck=key(row['prompt']),key(row['completion'])
        if pk in used_prompt or ck in used_reply or row['post_id'] in used_post:return False
        batch.append(row);used_prompt.add(pk);used_reply.add(ck);used_post.add(row['post_id'])
        return True
    for cat,part in quotas.items():
        for r in by_cat.get(cat,[])[:max(round(args.limit*part)*3,1)]:
            if sum(x['category']==cat for x in batch)>=round(args.limit*part):break
            add(r)
    all_rest=sorted((r for lst in by_cat.values() for r in lst),key=lambda x:(-x['score'],x['id']))
    for row in all_rest:
        if len(batch)>=args.limit:break
        add(row)
    batch.sort(key=lambda r:(-r['score'],r['id']))
    csv_write(EXPAND,batch,list(batch[0].keys()) if batch else [
        'approve','id','score','category','game','origin','post_id','source_url','context_signals',
        'review_warning','prompt','completion','v41_review','v41_reviewer_note','v41_reference_checked'])
    print('源文件读取:',all_counts['read'],'| 新样本符合规则:',all_counts['eligible_new'])
    print('新候选输出:',len(batch),'| 类别:',dict(Counter(r['category'] for r in batch)))
    print('注意：新候选 approve 为空，需经过语境审核；绝不可直接合并到旧训练集。')


def main():
    p=argparse.ArgumentParser(description='QWbot V4.1 数据质量审计与原始语料扩充（纯标准库，不训练）')
    sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('audit',help='分析160条已入选语料、推荐复审队列、输出独立测试题')
    a.add_argument('--priority-limit',type=int,default=180)
    a.set_defaults(func=audit)
    b=sub.add_parser('expand',help='从本机106k原始JSONL抽取新的待审样本')
    b.add_argument('--limit',type=int,default=800)
    b.add_argument('--min-score',type=int,default=5)
    b.set_defaults(func=expand)
    args=p.parse_args()
    if getattr(args,'limit',1)<1 or getattr(args,'priority_limit',1)<1:p.error('数量必须大于0')
    args.func(args)

if __name__=='__main__':
    main()
