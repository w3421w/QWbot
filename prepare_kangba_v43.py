#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""QWbot V4.3 抗吧语料扩充与严格构建（仅 Python 标准库，CPU可运行）。

1. python prepare_kangba_v43.py prepare
2. 审查 datasets/kangba_v43_review_queue.csv 的 approve 列： y/n/空白
3. python prepare_kangba_v43.py build
可选：python prepare_kangba_v43.py expand --limit 800
     （从原本106k JSONL读取新论坛候选，绝不自动批准）

不会修改任何 V2/V3/V4 LoRA、旧训练集或原始贴吧文件。
"""
from __future__ import annotations

import argparse
import csv
import difflib
import hashlib
import json
import random
import re
import shutil
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'datasets'
EXTRAS = ROOT / 'v43_inputs'
OLD_FORUM = 'kangba_v4_kangba_candidates.csv'
OLD_SEEDS = 'kangba_v4_kangba_seed_examples.csv'
PRIORITY = 'kangba_v41_recheck_priority.csv'
AUTHORED = 'kangba_v43_authored_examples.csv'
HOLDOUT = 'kangba_v41_new_holdout.csv'
NEW_HOLDOUT = 'kangba_v43_holdout.csv'
QUEUE = DATA / 'kangba_v43_review_queue.csv'
EXTENDED = DATA / 'kangba_v43_new_forum_candidates.csv'
TRAIN = DATA / 'kangba_v43_train.jsonl'
VALID = DATA / 'kangba_v43_valid.jsonl'
META = DATA / 'kangba_v43_selected_metadata.csv'
REPORT = DATA / 'kangba_v43_report.json'

SYSTEM = (
    '你是QQ群里长期混抗压背锅吧的普通群友，不是客服。先听懂对方在说什么，再自然接话。'
    '可以用反串、回旋镖、阴阳怪气，也可以自然使用“典”“绷”“闹麻了”等短梗；'
    '该用才用，不要随机拼接黑话、硬套游戏话题或无意义复读。'
    '一般一句话就够，不强制字数；认真求助和明确不想被开玩笑时正常回应。'
    '新闻、赛事、转会等事实未知时，不编造、不假装已查过。只输出要发到QQ群的回复。'
)
FIELDS = [
    'approve', 'id', 'category', 'game', 'origin', 'post_id', 'source_url',
    'prompt', 'completion', 'review_reason', 'review_level', 'v43_tags',
    'prior_approve', 'source_score', 'v43_warning',
]
CATEGORIES = ['贷款吹与反串', '回旋镖与清算', '粉丝争论与拆台', '抽象短梗',
              '日常水帖接话', '赛事梗与讨论', '事实与边界', '认真回应', '多轮群聊接梗']

# 只用于筛选漏网风险，不判定真实语义。对"典""绷"等不做最短字数限制。
PRIVATE = re.compile(r'(?<!\d)1[3-9]\d{9}(?!\d)|[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}|(?:微信|QQ号|手机号|住址)\s*[:：=]\s*\S{5,}', re.I)
AD = re.compile(r'刷单|代练接单|扫码.{0,6}领|加盟代理|优惠券|推广邀请码|加(?:我)?(?:微信|vx)(?:领|买)|https?://|www\.', re.I)
SEVERE = re.compile(r'杀你全家|死全家|开盒.*(?:住址|电话)|人肉搜索|种族灭绝|父母双亡|死妈|\bnmsl\b', re.I)
FACT_IN_REPLY = re.compile(r'(?:(?:已|刚|昨天|今天|今晚).{0,6}(?:官宣|转会|退役|处罚|被抓|确诊))|(?:据可靠消息|内部消息|实锤|曝出|签约已定)', re.I)
TOPIC_TERMS = re.compile(r'(?i)\b(?:lpl|lck|tes|blg|t1|rng|ig|jdg|faker|elk|bin|jkl)\b|战队|选手|比赛|冠军|联赛|赛区|超话|粉丝|抗吧|抗抗|串子|贴吧')
STYLE_HINT = re.compile(r'闹麻|典|绷|贷款|嘴硬|清算|回旋镖|双标|抽象|是吧|急了|开香槟|纯路人')


def tidy(v):
    return unicodedata.normalize('NFKC', str(v or '')).strip()


def norm(v):
    return re.sub(r'[^\w\u4e00-\u9fff]', '', tidy(v)).casefold()


def digest(v):
    return hashlib.sha1(v.encode('utf-8')).hexdigest()[:20]


def locate(name, required=True):
    # 用户目录中的新文件优先，工具包内的备份输入只是免去重复复制。
    for p in [DATA / name, EXTRAS / name]:
        if p.exists():
            return p
    if required:
        raise FileNotFoundError('缺少 ' + name + '。把工具包 v43_inputs 文件夹一起复制过去。')
    return None


def restore_csv_cell(v):
    v = v or ''
    return v[1:] if v.startswith("'") and len(v) > 1 and v[1] in '=+-@\t\r' else v


def readcsv(p):
    with p.open(encoding='utf-8-sig', newline='') as f:
        return [{k:restore_csv_cell(v) for k,v in row.items()} for row in csv.DictReader(f)]


def writecsv(p, rows, fields=None, overwrite=False):
    if p.exists() and not overwrite:
        raise FileExistsError(f'{p} 已存在，先备份审核结果，或显式加 --overwrite')
    p.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or FIELDS
    tmp = p.with_suffix(p.suffix + '.tmp')
    with tmp.open('w', encoding='utf-8-sig', newline='') as f:
        w=csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            # 防止CSV在Excel中被当成公式执行，读取时会恢复。
            record={}
            for k in fields:
                v=str(r.get(k,'') or '')
                if v.lstrip().startswith(('=','+','-','@','\t','\r')):
                    v="'"+v
                record[k]=v
            w.writerow(record)
    tmp.replace(p)
    print('写入:',p,'|',len(rows),'行')


def holdouts():
    p=locate(HOLDOUT,False)
    ans=[]
    if p:
        ans=[r.get('prompt','') for r in readcsv(p)]
    fresh=locate(NEW_HOLDOUT,False)
    if fresh:ans += [r.get('prompt','') for r in readcsv(fresh)]
    # 之前V4、V2等已经使用过的旧测试题，也要排除。
    ans += [
      '我上单0/8，这把全怪打野','BLG今年绝对夺冠','这选手赛前吹上天，赛后直接隐身',
      '我LOL已经十连败了','今天不想打游戏了，感觉好累','你们抗抗就知道串',
      '昨天说绝不熬夜，今天又通宵了','我宣布这个新队已经是冠军了',
      '我瓦洛兰特队友天天干拉送人头','我们队才赢第一局，冠军已经稳了',
      '这个选手是不是昨天转会了？',
    ]
    return [s for s in ans if s]


def make_checks():
    src=holdouts()
    return (set(map(norm,src)), [(norm(x),x) for x in src])


def holdout_issue(prompt, checks):
    p=norm(prompt)
    if p in checks[0]: return '与留出测试题完全重合'
    if len(p)<=10:return ''
    # 快速的长度上界预筛，再对极少数相近长度候选做字符相似判断。
    for t,original in checks[1]:
        if len(t)<=10 or 2*min(len(p),len(t))/(len(p)+len(t))<0.82:
            continue
        if difflib.SequenceMatcher(None,p,t).ratio()>=0.82:
            return '与留出测试题高度近似'
    return ''


def risk_for(row):
    p,c=tidy(row.get('prompt')),tidy(row.get('completion'))
    if not p or not c:return '空白问答'
    if norm(p)==norm(c):return '原话照抄'
    if PRIVATE.search(p+'\n'+c):return '疑似联系方式/个人信息'
    if AD.search(p+'\n'+c):return '广告/外链'
    if SEVERE.search(p+'\n'+c):return '恶意威胁/严重辱骂'
    if FACT_IN_REPLY.search(c):return '回复存在未经核实的事实断言'
    if len(p)>500 or len(c)>450:return '过长/疑似脏数据'
    return ''


def make_review_row(r,origin=None,approval=None):
    out={k: r.get(k,'') or '' for k in FIELDS}
    out['origin']=origin or out['origin'] or 'forum_candidate'
    out['id']=out['id'] or 'new_'+digest(norm(out['prompt'])+'\n'+norm(out['completion']))
    if approval is not None:out['approve']=approval
    return out


def prepare(args):
    prev=readcsv(locate(PRIORITY))
    authored=readcsv(locate(AUTHORED))
    rows=[];seen=set()
    for r in authored:
        row=make_review_row(r,origin='authored_example_v43',approval=r.get('approve','y'))
        row['prior_approve']=''
        row['review_level']='原创示范；非真实贴吧回复；可手动改 approve=n'
        pair=(norm(row['prompt']),norm(row['completion']))
        if pair not in seen: rows.append(row);seen.add(pair)
    for r in prev:
        row=make_review_row(r,origin='forum_recheck_v43',approval='')
        row['prior_approve']=r.get('approve','')
        row['source_score']=r.get('score','')
        row['review_level']='此前暂缓：待重新检查语境、楼层、历史事实；不自动批准'
        row['v43_warning']='论坛缺完整楼中楼父评论；原评分只是关键词粗排'
        pair=(norm(row['prompt']),norm(row['completion']))
        if pair not in seen: rows.append(row);seen.add(pair)
    writecsv(QUEUE,rows,overwrite=args.overwrite)
    print('已预批准原创:',sum(r['approve'].lower()=='y' for r in rows if r['origin'].startswith('authored_')),
          '| 待复核真实论坛:',sum(not r['approve'].strip() and r['origin']=='forum_recheck_v43' for r in rows))
    print('只需改动审核表approve列。n/空白不会参与训练；原V4训练文件不修改。')


def expand(args):
    """从本机完整训练源提取新论坛候选；仅制作待审核CSV，不自动批准。"""
    source=DATA/'kangba_train.jsonl'
    if not source.exists():raise FileNotFoundError(f'扩充需要{source}，当前只可先执行 prepare / build')
    existing=[]
    for name in [OLD_FORUM,OLD_SEEDS,PRIORITY,AUTHORED]:
        p=locate(name,False)
        if p:existing.extend(readcsv(p))
    if QUEUE.exists():existing.extend(readcsv(QUEUE))
    used_pair={(norm(r.get('prompt')),norm(r.get('completion'))) for r in existing}
    used_prompt={norm(r.get('prompt')) for r in existing}
    checks=make_checks()
    eligible=[]; counts=Counter();seen=set()
    for line in source.open(encoding='utf-8'):
        if not line.strip():continue
        counts['source_lines']+=1
        try:
            j=json.loads(line)
            p='\n'.join(m['content'] for m in j['prompt'] if m['role']=='user')
            a=j['completion'][-1]['content']
        except (KeyError,TypeError,IndexError,ValueError):
            counts['bad_json']+=1;continue
        p,a=tidy(p),tidy(a)
        pk,ak=norm(p),norm(a)
        if (pk,ak) in used_pair or (pk,ak) in seen or pk in used_prompt:
            counts['old_or_duplicate']+=1;continue
        if not (5<=len(p)<=200 and 1<=len(a)<=70):
            counts['length_outlier']+=1;continue
        row={'prompt':p,'completion':a}
        if risk_for(row):counts['risk_excluded']+=1;continue
        if holdout_issue(p,checks):counts['test_leak_excluded']+=1;continue
        # 启发式只做候选排序，不自动推断关联性。
        pp={p[i:i+2] for i in range(len(p)-1) if '\u4e00'<=p[i]<='\u9fff'}
        aa={a[i:i+2] for i in range(len(a)-1) if '\u4e00'<=a[i]<='\u9fff'}
        shared=len(pp&aa)
        score=(2*min(shared,4))+(3 if 2<=len(a)<=28 else 1)+(2 if STYLE_HINT.search(a) else 0)
        if a in {'典','绷','闹麻了','急了'}:
            score+=2 # 短梗可入围，但有上下文要求，仍须审核
        cat=('粉丝争论与拆台' if ('粉丝' in p or '双标' in p or '饭圈' in p) else
             '回旋镖与清算' if re.search(r'之前|昨天|当初|以前|赛前|赛后|结果|反过来|回旋镖',p) else
             '贷款吹与反串' if re.search(r'贷款|夺冠|世一|第一|无敌|必赢|稳赢|冠军',p) else
             '抽象短梗' if len(a)<=10 or STYLE_HINT.search(a) else
             '赛事梗与讨论' if TOPIC_TERMS.search(p) else '日常水帖接话')
        if score < args.min_score:counts['low_score']+=1;continue
        uid='v43_forum_'+digest(pk+'\n'+ak)
        ent=make_review_row({'approve':'','id':uid,'category':cat,'game':'GENERAL',
            'origin':'forum_expanded_v43','post_id':'prompt_'+digest(pk),
            'prompt':p,'completion':a,'source_score':str(score),
            'review_level':'自动挖掘，未经逐条语义审核；不得直接训练',
            'v43_warning':'缺楼中楼结构；共享词片不代表正确配对；事实未经核验'})
        eligible.append(ent);seen.add((pk,ak));counts['eligible']+=1
    rng=random.Random(3407)
    rng.shuffle(eligible)
    eligible.sort(key=lambda r:(-int(r['source_score']),r['id']))
    target={ '贷款吹与反串':.18,'回旋镖与清算':.20,'粉丝争论与拆台':.22,
             '抽象短梗':.21,'日常水帖接话':.14,'赛事梗与讨论':.05}
    groups=defaultdict(list)
    for r in eligible:groups[r['category']].append(r)
    final=[];up=set();ua=set()
    def add(r):
        p,a=norm(r['prompt']),norm(r['completion'])
        if p in up or a in ua:return False
        final.append(r);up.add(p);ua.add(a);return True
    for cat,fraction in target.items():
        for r in groups.get(cat,[]):
            if len(final)>=args.limit:break
            if sum(x['category']==cat for x in final)>=round(fraction*args.limit):break
            add(r)
    for r in eligible:
        if len(final)>=args.limit:break
        add(r)
    writecsv(EXTENDED,final,overwrite=args.overwrite)
    print('源JSONL记录:',counts['source_lines'],'| 合规则新候选:',counts['eligible'],
          '| 待人工审核输出:',len(final),'| 本命令自动批准:0')
    print('要加入V4.3审核队列，请把新CSV中经过核对的approve改为y，然后运行 build。')


class DSU:
    def __init__(self,n):self.parent=list(range(n))
    def find(self,a):
        while self.parent[a]!=a:
            self.parent[a]=self.parent[self.parent[a]];a=self.parent[a]
        return a
    def union(self,a,b):self.parent[self.find(a)]=self.find(b)


def split_groups(rows,fraction,seed):
    # 按原帖、问题、答案建立连通分组，避免跨集合共享相同原帖/问法/回复。
    dsu=DSU(len(rows));seen_post={};seen_p={};seen_a={}
    for i,r in enumerate(rows):
        keys=[(seen_p,norm(r['prompt'])),(seen_a,norm(r['completion']))]
        if r['origin'].startswith('forum') and r.get('post_id'):
            keys.append((seen_post,r['post_id']))
        for d,k in keys:
            if k in d:dsu.union(i,d[k])
            else:d[k]=i
    groups=defaultdict(list)
    for i,r in enumerate(rows):groups[dsu.find(i)].append(r)
    rng=random.Random(seed);batches=list(groups.values());rng.shuffle(batches)
    want=max(1,round(len(rows)*fraction))
    valid=[];train=[]
    for batch in batches:
        if len(valid)<want and len(rows)-len(valid)-len(batch)>=1:valid.extend(batch)
        else:train.extend(batch)
    if not valid or not train:raise ValueError('无法分出非空训练/验证集，请补充更多独立话题')
    return train,valid


def build(args):
    if not QUEUE.exists():raise FileNotFoundError('先运行 python prepare_kangba_v43.py prepare')
    old_forum=readcsv(locate(OLD_FORUM))
    old_seed=readcsv(locate(OLD_SEEDS))
    # 避免误把本地未审核的1000行旧CSV当成已审核版继续构建。
    forum_y=sum((r.get('approve','') or '').lower().strip()=='y' for r in old_forum)
    if forum_y<30:
        raise ValueError(f'旧论坛CSV中仅有{forum_y}条approve=y；请确认 datasets/{OLD_FORUM} 是以前105条入选的审核版，不是extract初始候选表')
    olds=old_forum+old_seed
    new=readcsv(QUEUE)
    if EXTENDED.exists():new.extend(readcsv(EXTENDED))
    checks=make_checks()
    all_rows=[];counts=Counter();seen_ids=set();seen_pairs=set();reply_counts=Counter()
    for r in olds+new:
        if (r.get('approve','') or '').strip().lower()!='y':
            counts['not_approved']+=1;continue
        row=make_review_row(r)
        ident=row['id'];p=row['prompt'];a=row['completion']
        if ident in seen_ids:counts['id_duplicate']+=1;continue
        seen_ids.add(ident)
        risk=risk_for(row)
        if risk:
            counts['risk_removed_'+risk]+=1;continue
        leak=holdout_issue(p,checks)
        if leak:
            counts['test_leak_removed']+=1;continue
        pair=(norm(p),norm(a))
        if pair in seen_pairs:
            counts['pair_duplicate']+=1;continue
        if reply_counts[norm(a)]>=args.max_per_reply:
            counts['reply_frequency_cap']+=1;continue
        # 新挖掘论坛候选必须显式审核Y才能加入；不能用auto-top。
        all_rows.append(row)
        seen_pairs.add(pair);reply_counts[norm(a)]+=1
        counts['accepted_'+row['origin']]+=1
    if len(all_rows)<30:
        raise ValueError(f'只有{len(all_rows)}条合格样本；请检查CSV的approve和风险标记')
    train,valid=split_groups(all_rows,args.valid_fraction,args.seed)
    assert not ({norm(r['prompt']) for r in train}&{norm(r['prompt']) for r in valid})
    assert not ({norm(r['completion']) for r in train}&{norm(r['completion']) for r in valid})
    assert not ({r['post_id'] for r in train if r['origin'].startswith('forum')}
                &{r['post_id'] for r in valid if r['origin'].startswith('forum')})
    def conv(r):
        return {'prompt':[{'role':'system','content':SYSTEM},
                          {'role':'user','content':r['prompt']}],
                'completion':[{'role':'assistant','content':r['completion']}]}
    for path,items in [(TRAIN,train),(VALID,valid)]:
        if path.exists() and not args.overwrite:raise FileExistsError(f'{path} 已存在，增加 --overwrite 才会覆盖V4.3输出')
        path.parent.mkdir(parents=True,exist_ok=True)
        tmp=path.with_suffix('.tmp')
        with tmp.open('w',encoding='utf-8') as f:
            for r in items:f.write(json.dumps(conv(r),ensure_ascii=False)+'\n')
        tmp.replace(path)
    writecsv(META,[dict(r,split='train') for r in train]+[dict(r,split='valid') for r in valid],FIELDS+['split'],args.overwrite)
    report={
        'version':'V4.3','train_count':len(train),'valid_count':len(valid),
        'category_distribution':dict(Counter(r['category'] for r in all_rows)),
        'origins':dict(Counter(r['origin'] for r in all_rows)),
        'mean_answer_chars':round(sum(len(r['completion']) for r in all_rows)/len(all_rows),2),
        'one_to_three_chars':sum(len(r['completion'])<=3 for r in all_rows),
        'diagnostics':dict(counts),'holdout_count':len(checks[1]),
        'train_path':str(TRAIN),'valid_path':str(VALID),
        'limitations':[
          '自动提取的论坛回复缺少真实楼中楼父评论；approve=y仍不能证明事实正确。',
          '原创示范非真实抗吧评论，可能造成表达模式单一；请比较实际模型回复。',
          '仅做字面高度相似的测试污染过滤；语义近重复仍需人工核查。',
          '训练、验证分组按原帖/问题/回答去重，但小规模验证Loss波动较大。'
        ]}
    if REPORT.exists() and not args.overwrite:raise FileExistsError(f'{REPORT} 已存在；加 --overwrite 仅覆盖V4.3文件')
    REPORT.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('========== V4.3构建完成 =========')
    print('训练:',len(train),'验证:',len(valid),'| 不同来源:',report['origins'])
    print('场景:',report['category_distribution'])
    print('1~3字短梗:',report['one_to_three_chars'],'| 平均回复长度:',report['mean_answer_chars'])
    print('跳过统计:',dict(counts))
    print('数据报告:',REPORT)
    print('尚未启动GPU训练。请先查看样本，并运行 train_kangba_v43.py --check-only。')


def main():
    parser=argparse.ArgumentParser(description='QWbot V4.3 抗吧语料数据建设')
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare',help='整合144条新原创示范+180条既有暂缓论坛候选，生成可编辑审核队列')
    p.add_argument('--overwrite',action='store_true',help='显式覆盖V4.3输出（不会改旧版）')
    p.set_defaults(func=prepare)
    p=sub.add_parser('expand',help='从本机已有106k条JSONL提取新候选，默认全部未批准')
    p.add_argument('--limit',type=int,default=800)
    p.add_argument('--min-score',type=int,default=5)
    p.add_argument('--overwrite',action='store_true')
    p.set_defaults(func=expand)
    p=sub.add_parser('build',help='仅纳入approve=y，并保持测试题与数据集隔离')
    p.add_argument('--valid-fraction',type=float,default=.12)
    p.add_argument('--seed',type=int,default=3407)
    p.add_argument('--max-per-reply',type=int,default=2)
    p.add_argument('--overwrite',action='store_true')
    p.set_defaults(func=build)
    args=parser.parse_args()
    if hasattr(args,'limit') and args.limit<1:parser.error('--limit必须大于0')
    if hasattr(args,'valid_fraction') and not 0<args.valid_fraction<.5:parser.error('valid-fraction须在0~0.5之间')
    args.func(args)

if __name__=='__main__':main()
