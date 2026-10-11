#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V4.1匿名A/B模型对照：原始Qwen3 vs 抗吧V4（不训练，不修改LoRA）。
先运行 audit_kangba_v41.py audit 生成新的独立测试集。
使用:
  python compare_kangba_v41.py --limit 12 --repeats 2
  python compare_kangba_v41.py --limit 0 --repeats 2
输出匿名评分CSV、解盲映射JSON、原始结果JSON；不要把测试题或模型回复用于训练。
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
import random
from pathlib import Path

ROOT=Path(__file__).resolve().parent
os.environ.setdefault('HF_ENDPOINT','https://hf-mirror.com')
os.environ.setdefault('HF_HOME',str(ROOT/'hf_cache'))
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')

# Unsloth必须先于torch和peft导入
import unsloth  # noqa: F401
import torch
from peft import PeftModel
from unsloth import FastModel

BASE='unsloth/Qwen3-1.7B-unsloth-bnb-4bit'
V4=ROOT/'outputs'/'kangba_v4_lora'
TESTS=ROOT/'datasets'/'kangba_v41_new_holdout.csv'
OUT=ROOT/'outputs'
SYSTEM=(
    '你是QQ群里的普通群友，说话带抗压背锅吧那种简短、抽象、反串、回旋镖的味道。'
    '先理解对方在说什么，接得上话才整活；不用客服腔，不长篇说教，不硬塞游戏梗。'
    '抗吧常聊电竞但不限电竞，日常水帖也能正常聊；真实事件不编造，严肃求助时正常回应。'
)


def main():
    p=argparse.ArgumentParser(description='抗吧V4.1新测试题：匿名A/B主观评价')
    p.add_argument('--limit',type=int,default=12,help='首次建议12，0表示全部测试题')
    p.add_argument('--repeats',type=int,default=2,help='每题重复采样次数，默认2')
    p.add_argument('--max-new-tokens',type=int,default=48)
    p.add_argument('--temperature',type=float,default=.8)
    args=p.parse_args()
    if args.limit<0 or not 1<=args.repeats<=5 or args.max_new_tokens<1 or args.temperature<=0:
        p.error('参数无效：limit>=0，repeats在1-5，max_new_tokens>0，temperature>0')
    if not (V4/'adapter_config.json').exists():
        raise FileNotFoundError('未找到V4 LoRA：'+str(V4/'adapter_config.json'))
    if not TESTS.exists():
        raise FileNotFoundError('缺少测试题：'+str(TESTS)+'\n先运行 python audit_kangba_v41.py audit')
    with TESTS.open(encoding='utf-8-sig',newline='') as f:
        questions=list(csv.DictReader(f))
    # 分层交错取题：--limit 12时覆盖7个不同类型，而不是只测列表前两个类别。
    by_cat={}
    for item in questions:
        by_cat.setdefault(item.get('category','其他'), []).append(item)
    balanced=[]
    for j in range(max(len(arr) for arr in by_cat.values())):
        for arr in by_cat.values():
            if j<len(arr):balanced.append(arr[j])
    questions=balanced
    if args.limit: questions=questions[:args.limit]
    if not questions:raise ValueError('测试题CSV为空')
    if not torch.cuda.is_available():raise RuntimeError('未检测到CUDA；请在QWbot GPU环境执行')
    print('载入原始Qwen3 1.7B和V4 LoRA……',flush=True)
    model,tokenizer=FastModel.from_pretrained(model_name=BASE,max_seq_length=512,
                                               dtype=torch.float16,load_in_4bit=True)
    model=PeftModel.from_pretrained(model,str(V4),adapter_name='kangba_v4')
    model.eval()
    im_end=tokenizer.convert_tokens_to_ids('<|im_end|>')
    eos=tokenizer.eos_token_id
    stopids=list(dict.fromkeys(i for i in [eos,im_end] if isinstance(i,int) and i>=0))
    pad=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos
    if pad is None:pad=stopids[0]

    def generate_one(prompt,base,seed):
        messages=[{'role':'system','content':SYSTEM},{'role':'user','content':prompt}]
        ids=tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,
                        enable_thinking=False,return_tensors='pt').to('cuda')
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        params={'input_ids':ids,'max_length':ids.shape[-1]+args.max_new_tokens,
                'do_sample':True,'temperature':args.temperature,'top_p':.9,
                'repetition_penalty':1.15,'no_repeat_ngram_size':4,
                'eos_token_id':stopids,'pad_token_id':pad}
        with torch.inference_mode():
            if base:
                with model.disable_adapter(): result=model.generate(**params)
            else:
                model.set_adapter('kangba_v4')
                result=model.generate(**params)
        return tokenizer.decode(result[0,ids.shape[-1]:],skip_special_tokens=True).strip() or '（空回复）'

    # 盲测文件用A/B，映射另存，不在CSV里透露模型身份。
    records, keys, raw=[],[],[]
    for index,q in enumerate(questions,1):
        question=q['prompt']
        for trial in range(1,args.repeats+1):
            sample_seed=3407+index*101+trial
            a0=generate_one(question,True,sample_seed)
            a4=generate_one(question,False,sample_seed)
            rid=f'T{index:03d}-R{trial}'
            # 按确定性哈希打乱顺序；仅密钥文件里保存对应模型。
            swap=int(hashlib.sha256(rid.encode()).hexdigest(),16)%2==0
            ans_a,ans_b=(a4,a0) if swap else (a0,a4)
            keys.append({'id':rid,'A':'V4' if swap else '原始Qwen',
                         'B':'原始Qwen' if swap else 'V4'})
            records.append({'id':rid,'category':q.get('category',''),'prompt':question,
                'reply_A':ans_a,'reply_B':ans_b,
                'which_is_better_A_B_tie':'',
                'relevance_A_0_2':'','relevance_B_0_2':'',
                'style_A_0_2':'','style_B_0_2':'',
                'coherence_A_0_2':'','coherence_B_0_2':'',
                'fabrication_or_harm_A_y_n':'','fabrication_or_harm_B_y_n':'',
                'notes':''})
            raw.append({'id':rid,'category':q.get('category',''),
                        'prompt':question,'original_qwen':a0,'kangba_v4':a4})
            print(f'{rid} {q.get("category","")} | {question}',flush=True)
            print(' A:',ans_a[:180],flush=True)
            print(' B:',ans_b[:180],flush=True)
    OUT.mkdir(parents=True,exist_ok=True)
    files=[OUT/'kangba_v41_blind_scoring.csv',OUT/'kangba_v41_blind_key.json',
           OUT/'kangba_v41_raw_results.json']
    for f in files:
        if f.exists():raise FileExistsError('为避免覆盖已有评分，输出文件已存在，请先备份或重命名：'+str(f))
    with files[0].open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    files[1].write_text(json.dumps({'purpose':'解盲专用；先填写CSV评分，最后才查看本文件',
                                    'mapping':keys},ensure_ascii=False,indent=2),encoding='utf-8')
    files[2].write_text(json.dumps({'system_prompt':SYSTEM,'params':vars(args),
                                    'results':raw},ensure_ascii=False,indent=2),encoding='utf-8')
    print('\n匿名评价表:',files[0], '\n解盲映射:',files[1], '\n原始结果:',files[2])
    print('先在评分CSV中填A/B/Tie和各维度0-2分，再看映射；测试题及生成内容不得进入训练集。')

if __name__=='__main__':
    main()
