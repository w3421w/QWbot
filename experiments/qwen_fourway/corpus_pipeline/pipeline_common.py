#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared, dependency-free helpers for the read-only Kangba corpus pipeline."""
from __future__ import annotations
import csv, hashlib, json, re, unicodedata
from pathlib import Path
from difflib import SequenceMatcher

ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = ROOT / 'datasets' / 'kangba_raw' / 'data'
TEST_FILE = ROOT / 'datasets' / 'QWbot_V44_真实群聊影子测试_匿名31题.jsonl'
FIELDS = ['id','source','conversation_id','context','incoming_message','target_reply','source_post_id','source_reply_id','source_file','reply_index','reply_relation','quality_flags','quality_score','provenance_type','review_status','license_status','text_hash','near_duplicate_of']
AD = re.compile(r'购物礼金|优惠礼包|领红包|返现(?:活动|领取)|刷单|兼职日结|推广邀请码|扫码(?:进群|购买|领取|加群)|加(?:我)?(?:微|v|vx)(?:信)?(?:领|买|拿|咨询|下单)|代练接单|陪玩接单|招商加盟|招代理|买粉|出售账号|联系(?:我)?(?:微信|QQ|vx)|京东首页搜|淘宝(?:店铺|优惠券)|拼多多(?:砍价|领券)|点击(?:链接|购买)|联系(?:客服|商家)下单|优惠券(?:领取|入口)', re.I)
LINK = re.compile(r'(?:https?://|www\.|t\.cn/|bit\.ly/|短链|vx[:：]|微信[:：]|QQ群[:：])', re.I)
AD_CTX = re.compile(r'购买|优惠|下单|私聊|推广|扫码|返现|免费领|加群|客服|福利|赚米|包邮|直降|出售', re.I)
THREAT = re.compile(r'杀你全家|死全家|人肉搜索(?:地址|电话)|开盒(?:地址|家人)|种族灭绝', re.I)
PHONE = re.compile(r'(?<!\d)1[3-9]\d{9}(?!\d)')
EMAIL = re.compile(r'(?<![\w.-])[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}(?![\w.-])')
CHAT = re.compile(r'(?:(?:微信|vx|V信|QQ号|QQ号码)\s*[:：=]\s*)[a-zA-Z0-9_-]{5,}', re.I)
LOW = re.compile(r'^(?:[哈啊呃哦嗯啧哎]{3,}|666+|\d+|顶|插眼|蹲|前排|cy|dd|mark|确实)$', re.I)

def norm(v):
    if not isinstance(v, str): return ''
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', v)).strip()

def redact(s):
    return CHAT.sub('[联系方式已脱敏]', EMAIL.sub('[邮箱已脱敏]', PHONE.sub('[手机号已脱敏]', s)))

def post_id(url, path, index):
    m = re.search(r'/p/(\d+)', url or '')
    return m.group(1) if m else 'post-' + hashlib.sha1(f'{path}:{index}:{url}'.encode()).hexdigest()[:16]

def text_hash(s):
    return hashlib.sha256(norm(s).casefold().encode('utf-8')).hexdigest()

def key(s):
    return re.sub(r'\W+', '', norm(s).casefold(), flags=re.UNICODE)

def classify(text):
    if not text: return 'empty'
    if AD.search(text) or (LINK.search(text) and AD_CTX.search(text)): return 'advertisement'
    if THREAT.search(text): return 'explicit_threat'
    return None

def natural_score(text):
    if not text: return 0.0
    score = 0.4
    if len(text) >= 6: score += 0.2
    if not LOW.fullmatch(text): score += 0.2
    if any(c in text for c in '，。！？,.!?'): score += 0.1
    if len(text) > 300: score -= 0.1
    return round(max(0, min(1, score)), 2)

def relation_score():
    return 0.0  # source contains no relation metadata; never infer one

def read_jsonl(path):
    with path.open(encoding='utf-8-sig') as f:
        for line_no,line in enumerate(f,1):
            try: yield line_no,json.loads(line)
            except json.JSONDecodeError: continue

def test_texts():
    vals=[]
    if not TEST_FILE.exists(): return vals
    for _,row in read_jsonl(TEST_FILE):
        if not isinstance(row,dict): continue
        for k in ('incoming_message','example_reply_for_human_review'):
            if row.get(k): vals.append(norm(row[k]))
        for c in row.get('context',[]) or []:
            if isinstance(c,dict) and c.get('text'): vals.append(norm(c['text']))
    return vals

def near(a,b):
    if not a or not b: return False
    if a == b: return True
    if len(a)<8 or len(b)<8: return False
    return SequenceMatcher(None,a,b).ratio() >= .92

def csv_write(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore'); w.writeheader()
        for r in rows:
            out={k:r.get(k,'') for k in fields}
            for k,v in out.items():
                if isinstance(v,str) and v.lstrip().startswith(('=','+','-','@')): out[k]="'"+v
            w.writerow(out)
