#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""QWbot V4.4：在已完成的V4.3 501条数据上扩充原创专项数据。

python prepare_kangba_v44.py prepare
python prepare_kangba_v44.py build
python train_kangba_v44.py --check-only

输入：
  datasets/kangba_v43_selected_metadata.csv （V4.3已生成，含固定train/valid列）
  v44_inputs/kangba_v44_authored_examples.csv （300条原创示例）
输出：
  datasets/kangba_v44_review.csv
  datasets/kangba_v44_train.jsonl
  datasets/kangba_v44_valid.jsonl
  datasets/kangba_v44_metadata.csv
  datasets/kangba_v44_report.json

不会修改V4.3数据、模型及旧评测文件。全部新原创示范均可更改approve=n排除。
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import random
import re
import shutil
import unicodedata
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'datasets'
INPUTS = ROOT / 'v44_inputs'
OLD = DATA / 'kangba_v43_selected_metadata.csv'
OLD_TRAIN = DATA / 'kangba_v43_train.jsonl'
OLD_VALID = DATA / 'kangba_v43_valid.jsonl'
SEED = INPUTS / 'kangba_v44_authored_examples.csv'
REVIEW = DATA / 'kangba_v44_review.csv'
V44_HOLDOUT = INPUTS / 'kangba_v44_holdout.csv'
V43_HOLDOUT = INPUTS / 'kangba_v43_holdout.csv'
V41_HOLDOUT = INPUTS / 'kangba_v41_new_holdout.csv'
OUTPUT_TRAIN = DATA / 'kangba_v44_train.jsonl'
OUTPUT_VALID = DATA / 'kangba_v44_valid.jsonl'
OUTPUT_META = DATA / 'kangba_v44_metadata.csv'
OUTPUT_REPORT = DATA / 'kangba_v44_report.json'

SYSTEM_FALLBACK = (
    '你是QQ群里长期混抗压背锅吧的普通群友，不是客服。先听懂对方在说什么，再自然接话。'
    '可以用反串、回旋镖、阴阳怪气，也可以自然使用“典”“绷”“闹麻了”等短梗；'
    '该用才用，不要随机拼接黑话、硬套游戏话题或无意义复读。'
    '一般一句话就够，不强制字数；认真求助和明确不想被开玩笑时正常回应。'
    '新闻、赛事、转会等事实未知时，不编造、不假装已查过。只输出要发到QQ群的回复。'
)


def normalize(s):
    return re.sub(r'[^\w\u4e00-\u9fff]', '', unicodedata.normalize('NFKC', str(s or ''))).casefold()


def load_csv(path):
    if not path.is_file():
        raise FileNotFoundError(f'缺少 {path}。请按README复制文件；不要重新生成V4.3数据。')
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fields,extrasaction='ignore');w.writeheader()
        for r in rows:
            # 电子表格公式注入防护：仅对以公式符号开头的文本加前导单引号。
            row={}
            for k in fields:
                s=str(r.get(k,'') or '')
                if s.lstrip().startswith(('=', '+', '-', '@', '\t', '\r')):
                    s="'"+s
                row[k]=s
            w.writerow(row)


def restore_formula_cell(s):
    s=str(s or '')
    return s[1:] if len(s)>1 and s[0]=="'" and s[1] in '=+-@\t\r' else s


def load_system():
    systems=set()
    for p in (OLD_TRAIN, OLD_VALID):
        if p.exists():
            with p.open('r',encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        systems.add(json.loads(line)['prompt'][0]['content'])
                        break
    if len(systems)>1:
        raise ValueError('V4.3训练/验证数据的系统提示词不一致，先检查旧数据。')
    return next(iter(systems)) if systems else SYSTEM_FALLBACK


def read_holdouts():
    xs=[]
    for path in (V44_HOLDOUT,V43_HOLDOUT,V41_HOLDOUT):
        if path.exists():
            xs.extend(r.get('prompt','') for r in load_csv(path))
    xs.extend([
        '我上单0/8，这把全怪打野','BLG今年绝对夺冠',
        '这选手赛前吹上天，赛后直接隐身','我LOL已经十连败了',
        '今天不想打游戏了，感觉好累','你们抗抗就知道串',
        '昨天说绝不熬夜，今天又通宵了','我宣布这个新队已经是冠军了',
        '这个选手是不是昨天转会了？',
    ])
    return [normalize(x) for x in xs if x.strip()]


def holdout_near(prompt, holdouts, threshold=0.90):
    x=normalize(prompt)
    if not x:return '空问题'
    for y in holdouts:
        if x==y:return '与测试题完全相同'
        if len(x)<10 or len(y)<10:continue
        if 2*min(len(x),len(y))/(len(x)+len(y)) < threshold:continue
        if SequenceMatcher(None,x,y).ratio()>=threshold:return '与测试题文本过于相似'
    return ''


def prepare(args):
    if REVIEW.exists() and not args.overwrite:
        raise FileExistsError(f'{REVIEW} 已存在。为了保留你的审核结果，请先备份；如确需重建加 --overwrite。')
    seeds=load_csv(SEED)
    if any('approve' not in x or 'prompt' not in x or 'completion' not in x for x in seeds):
        raise ValueError('新版示范CSV缺少approve/prompt/completion列')
    write_csv(REVIEW,seeds)
    print(f'写入可编辑审核表：{REVIEW}；原创示范{len(seeds)}条')
    print('默认预批准的全部是原创示范，不是真实贴吧评论；可以把不喜欢的样本approve改n。')


def make_example(r, system):
    return {
        'prompt':[{'role':'system','content':system},{'role':'user','content':r['prompt']}],
        'completion':[{'role':'assistant','content':r['completion']}],
    }


def build(args):
    if any(x.exists() for x in (OUTPUT_TRAIN,OUTPUT_VALID,OUTPUT_META,OUTPUT_REPORT)) and not args.overwrite:
        raise FileExistsError('V4.4输出已经存在，先备份；确定覆盖V4.4输出再加 --overwrite。不会影响V4.3。')
    old=load_csv(OLD)
    new=load_csv(REVIEW)
    if not old or not new:raise ValueError('V4.3元数据或V4.4审核表为空')
    if any(r.get('split') not in ('train','valid') for r in old):
        raise ValueError('旧数据必须有split=train/valid；请使用V4.3的kangba_v43_selected_metadata.csv')
    system=load_system()
    holdouts=read_holdouts()
    old_prompt={normalize(restore_formula_cell(r.get('prompt'))) for r in old}
    old_answer={normalize(restore_formula_cell(r.get('completion'))) for r in old}
    old_train_answers={normalize(restore_formula_cell(r.get('completion'))) for r in old if r['split']=='train'}
    old_valid_answers={normalize(restore_formula_cell(r.get('completion'))) for r in old if r['split']=='valid'}
    approved_short_memes={normalize(x) for x in ('典','绷','急了','纯路人是吧','典中典','闹麻了')}
    if '' in old_prompt or '' in old_answer:
        raise ValueError('V4.3旧数据含空白问答，不能继续构建')
    counts=Counter()
    new_selected=[]
    seen_prompt=set()
    seen_answer=set()
    for row in new:
        if str(row.get('approve','')).strip().lower() != 'y':
            counts['unapproved']+=1;continue
        r={k:restore_formula_cell(v).strip() for k,v in row.items()}
        p,a=r.get('prompt',''),r.get('completion','')
        pn,an=normalize(p),normalize(a)
        if not pn or not an or pn==an:
            counts['empty_or_echo']+=1;continue
        if len(p)>400 or len(a)>110:
            counts['too_long']+=1;continue
        if pn in old_prompt:
            counts['duplicate_prompt_against_v43']+=1;continue
        # 短梗可以针对不同情境训练2次；但不能让旧验证集答案泄漏进训练集。
        # 例如‘典’已在旧train出现，可新增另一语境；‘闹麻了’若在旧valid则暂不新增。
        force_train = an in old_train_answers and an in approved_short_memes and an not in old_valid_answers
        if an in old_answer and not force_train:
            counts['duplicate_answer_against_v43']+=1;continue
        if force_train:
            r['force_train']='y'
            counts['allowed_short_meme_new_context']+=1
        if pn in seen_prompt or an in seen_answer:
            counts['duplicate_new_prompt_or_reply']+=1;continue
        risk=holdout_near(p,holdouts)
        if risk:
            counts['test_leakage_excluded']+=1;print('留出题排除：',p[:65]);continue
        if any(not r.get(k) for k in ('id','category','group_id','origin')):
            raise ValueError('V4.4审核CSV的id/category/group_id/origin不能为空')
        r['source_split']='new'
        new_selected.append(r)
        seen_prompt.add(pn);seen_answer.add(an)
    if len(new_selected)<50:raise ValueError('新增入选不足50条，请确认审核CSV与源文件')

    # 每个类别留出一个完整的五条同话题小组，防止完全相同的构造句跨集合。
    by_cat=defaultdict(lambda:defaultdict(list))
    for row in new_selected:by_cat[row['category']][row['group_id']].append(row)
    rng=random.Random(args.seed)
    valid_groups=set()
    for cat, groups in sorted(by_cat.items()):
        options=[gid for gid in sorted(groups) if not any(x.get('force_train')=='y' for x in groups[gid])]
        rng.shuffle(options)
        # 同类别不得整体分入验证集，保证新示范每类仍参与训练。
        if len(options)>1:
            valid_groups.add(options[0])
    new_train=[r for r in new_selected if r['group_id'] not in valid_groups]
    new_valid=[r for r in new_selected if r['group_id'] in valid_groups]
    train=[dict(r,source_split='v43_train') for r in old if r['split']=='train']+new_train
    valid=[dict(r,source_split='v43_valid') for r in old if r['split']=='valid']+new_valid
    if not train or not valid:raise ValueError('数据集无法正常划分')
    tp={normalize(restore_formula_cell(r['prompt'])) for r in train}
    vp={normalize(restore_formula_cell(r['prompt'])) for r in valid}
    ta={normalize(restore_formula_cell(r['completion'])) for r in train}
    va={normalize(restore_formula_cell(r['completion'])) for r in valid}
    if tp&vp or ta&va:
        raise ValueError('发现跨训练/验证集的完全相同问题或答案，构建终止')
    old_train_n=sum(r['split']=='train' for r in old)
    old_valid_n=sum(r['split']=='valid' for r in old)
    if sum(r['source_split']=='v43_train' for r in train)!=old_train_n or sum(r['source_split']=='v43_valid' for r in valid)!=old_valid_n:
        raise ValueError('V4.3原有训练/验证分配发生变化，停止')
    DATA.mkdir(exist_ok=True,parents=True)
    for path,rows in ((OUTPUT_TRAIN,train),(OUTPUT_VALID,valid)):
        tmp=path.with_suffix('.tmp')
        with tmp.open('w',encoding='utf-8') as f:
            for row in rows:
                obj=make_example({**row,'prompt':restore_formula_cell(row['prompt']),'completion':restore_formula_cell(row['completion'])},system)
                f.write(json.dumps(obj,ensure_ascii=False)+'\n')
        tmp.replace(path)
    metadata=[{**r,'split':'train'} for r in train]+[{**r,'split':'valid'} for r in valid]
    write_csv(OUTPUT_META,metadata)
    report={
        'version':'V4.4','system_prompt_same_as_v43':True,
        'v43_original_total':len(old),'v43_original_train':old_train_n,'v43_original_valid':old_valid_n,
        'new_input_rows':len(new),'new_approved_accepted':len(new_selected),
        'new_train':len(new_train),'new_valid':len(new_valid),
        'train_count':len(train),'valid_count':len(valid),'total_count':len(train)+len(valid),
        'category_new':dict(Counter(r['category'] for r in new_selected)),
        'new_reply_length_mean':round(sum(len(r['completion']) for r in new_selected)/len(new_selected),2),
        'new_short_reply_1to3_chars':sum(len(r['completion'])<=3 for r in new_selected),
        'new_multiturn_count':sum('群友A：' in r['prompt'] for r in new_selected),
        'skipped':dict(counts),
        'validation_group_ids':sorted(valid_groups),
        'limitations':[
            '300条新增内容是原创训练示范，不是抗吧真实帖子。',
            '词句重复检测无法保证消除语义近重复；应另外使用未进入训练的独立测试题。',
            '原有V4.3验证样本保留，新增验证组可能仍与训练数据存在话题层面的相似性。',
            '样本数增加不能替代真实群聊效果测评；不得凭Loss判断新模型优劣。',
        ],
    }
    OUTPUT_REPORT.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('========== V4.4构建成功 =========')
    print('旧V4.3固定：训练',old_train_n,'验证',old_valid_n)
    print('新V4.4加入：训练',len(new_train),'验证',len(new_valid))
    print('总计：训练',len(train),'验证',len(valid),'合计',len(train)+len(valid))
    print('新样本分类：',report['category_new'])
    print('跳过：',dict(counts))
    print('报告：',OUTPUT_REPORT)
    print('下一步：python train_kangba_v44.py --check-only')


def main():
    ap=argparse.ArgumentParser(description='QWbot V4.4专项抗吧语料合并、复核与构建')
    sub=ap.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare',help='生成可编辑300条原创样本审核表；默认不覆盖')
    p.add_argument('--overwrite',action='store_true')
    p.set_defaults(func=prepare)
    p=sub.add_parser('build',help='保留旧V4.3训练/验证划分，新增数据分组划分')
    p.add_argument('--overwrite',action='store_true')
    p.add_argument('--seed',type=int,default=3407)
    p.set_defaults(func=build)
    args=ap.parse_args();args.func(args)
if __name__=='__main__':main()
