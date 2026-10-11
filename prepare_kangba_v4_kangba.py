#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""QWbot V4：专注抗压背锅吧的语言风格，不限定LOL等游戏话题。
全量源文件只读 -> 抗吧风格候选 -> 人工审核 -> SFT JSONL.

用途（均在 ~/Qwen_Finetune 下运行）：
 python prepare_kangba_v4_kangba.py extract --limit 1000
 python prepare_kangba_v4_kangba.py build              # 仅 approve=y + 独立的原创示例
 python prepare_kangba_v4_kangba.py build --auto-top 250  # 未审核候选也取高分250条（仅实验）
 python prepare_kangba_v4_kangba.py stats

依赖：Python 标准库；不会修改原有 prepare_kangba_corpus.py、训练集、模型。
原始贴吧数据含用户内容和潜在事实错误；自动评分仅作候选粗排，绝非人工真实性校验。
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
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'datasets'
SOURCE = DATA / 'kangba_train.jsonl'
REVIEW_SOURCE = DATA / 'kangba_review.csv'
CANDIDATES = DATA / 'kangba_v4_kangba_candidates.csv'
SEEDS = DATA / 'kangba_v4_kangba_seed_examples.csv'
TRAIN = DATA / 'kangba_v4_kangba_train.jsonl'
VALID = DATA / 'kangba_v4_kangba_valid.jsonl'
META = DATA / 'kangba_v4_kangba_selected_metadata.csv'
REPORT = DATA / 'kangba_v4_kangba_report.json'
EVAL_PROMPTS = DATA / 'kangba_v4_kangba_holdout_prompts.txt'

SYSTEM = (
    '你是QQ群里的普通群友，说话带抗压背锅吧那种简短、抽象、反串、回旋镖的味道。'
    '先理解对方在说什么，接得上话才整活；不用客服腔，不长篇说教，不硬塞游戏梗。'
    '抗吧常聊电竞但不限电竞，日常水帖也能正常聊；真实事件不编造，严肃求助时正常回应。'
)
FIELDS = [
    'approve', 'id', 'score', 'category', 'game', 'context_signals', 'review_warning',
    'origin', 'post_id', 'source_url', 'prompt', 'completion'
]

LOL = re.compile(r'(?:英雄联盟|(?<![a-z0-9])(?:lol|lpl|lck|msi|rng|blg|jdg|tes|nip|lng|we|t1|faker|uzi|elk)(?![a-z0-9])|'
                 r'峡谷|打野|上单|中单|辅助|对线|大龙|小龙|补刀|野区|沙皇|巢虫|眼位|峡谷之巅|赛区)', re.I)
VAL = re.compile(r'(?:瓦洛兰特|无畏契约|(?<![a-z0-9])(?:valorant|vct|zmjjkk|smoggy|prx)(?![a-z0-9])|'
                 r'残局|干拉|补枪|强起|半起|起枪|架枪|爆头线|经济局|下包|拆包|特工|烟位|手枪局|存枪)', re.I)
GAME_ANCHORS = [
    'blg','edg','jdg','tes','ig','rng','nip','lng','t1','geng','gen','lpl','lck','msi','vct','prx',
    'faker','elk','bin','uzi','theshy','smoggy','zmjjkk','康康',
    '打野','上单','中单','下路','辅助','大龙','小龙','沙皇','对线','团战','排位','冠军',
    '赛前','赛后','世界赛','决赛','控龙','压线','连败','连胜','选手','队友',
    '干拉','补枪','残局','存枪','eco','强起','枪法','爆头','拆包','下包'
]
# 注意：队名+游戏语境才能判断EDG属于哪个游戏；仅EDG不自动赋值为LOL或VCT。
AMBIGUOUS_TEAM = re.compile(r'\b(?:edg|blg|fpx|jdg)\b', re.I)
STYLE = re.compile(r'反串|贷款|清算|回旋镖|闹麻|世一|体验卡|打脸|无敌|伟大|典|绷|幽默|'
                   r'就这|是吧|怎么说|不是哥们|真有你的|下饭|坐牢|纯路人|要不|'
                   r'全怪|这也能|倒反天罡|谁说|不然|还得是|急了|赢麻了|开香槟|'
                   r'谁问你了|别急|先质疑|享受|只需|被你|已经开始|名场面|又开始')
# 针对非电竞语境新增实体/论题线索。相同话题不等于正确配对，仍须人工审核。
KB_ANCHORS = [
    '抗吧','抗抗','串子','反串','贷款','清算','回旋镖','急了','典','绷','闹麻',
    '贴吧','吧友','水帖','楼主','网友','评论','流量','热度','饭圈','粉丝','黑粉',
    '嘴硬','打脸','双标','吹','洗','串','孝','骂','喷','夸','笑话','阴阳',
    '工资','上班','下班','加班','周末','放假','熬夜','睡觉','考试','学校','作业',
    '朋友','兄弟','群友','女朋友','男朋友','单身','恋爱','相亲','结婚',
    '手机','电脑','外卖','奶茶','火锅','食堂','游戏','视频','直播',
    '冠军','皮肤','战队','选手','教练','赛区','输赢','胜率','十连败',
]
REPLY_MEME = re.compile(r'闹麻|绷不住|典中典|经典|急了|贷款|回旋镖|打脸|清算|'
                        r'已经开始|先别急|体验卡|有一说一|是吧|赢麻了|还得是|'
                        r'什么含金量|不是哥们|串子|真有你的|谁问你了|就这|'
                        r'给你|你这|你们|那你|这么说|不然|建议|先把|直接|属于是')

BAD_LINK = re.compile(r'(?:https?://|www\.|t\.cn/|bit\.ly/)', re.I)
AD = re.compile(r'优惠礼包|购物礼金|扫码.{0,6}(领|加|买)|刷单|兼职日结|返现|优惠券|'
                r'招商加盟|推广邀请码|加(?:我)?(?:微信|vx)(?:领|买|下单)|京东首页搜|代练接单', re.I)
PERSONAL = re.compile(r'(?<!\d)1[3-9]\d{9}(?!\d)|[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}|'
                      r'(?:微信|QQ号|手机号|住址)\s*[:：=]\s*\S{5,}', re.I)
ABUSE = re.compile(r'杀你全家|死全家|开盒.{0,8}(地址|家人)|人肉搜索|种族灭绝|唐氏|弱智|\bNMSL\b|绝症粉丝|父母双亡|死妈', re.I)
UNVERIFIED_FACT = re.compile(r'官宣|转会|加盟|退役|被抓|假赛|合同|处罚|实锤|涉嫌|确诊|曝光|'
                             r'爆料|辟谣|内部消息|最新消息|据可靠消息|\b(?:breaking news)\b', re.I)
PROMPT_QUESTION = re.compile(r'谁|什么|哪|为什么|怎么|吗|？|\?')
EMOTION = re.compile(r'好累|难受|焦虑|抑郁|生病|手术|去世|压力大|崩溃|不想活|想哭')
STOP_BI = {
    '这个','那个','没有','有点','还是','就是','不是','什么','怎么','觉得','感觉','现在',
    '自己','我们','你们','他们','真的','还有','可以','应该','可能','的话','一个','一把',
    '比赛','因为','所以','时候','如果','已经','看看','现在','一下','然后','这样',
    '就算','今天','昨天','说的','一场','只能','就是','不行','不是'
}

# 这些是为QQ群精心编写的抗吧风格示范，不是采集到的贴吧评论。
# 系统将它们与论坛候选分开，后者全部默认未经审核。
SEED_EXAMPLES = [
    ('贷款吹与反串','GENERAL','这节目才第一期我已经预订年度最佳了','奖杯先焊上去是吧'),
    ('贷款吹与反串','GENERAL','这电影预告片一出我就宣布封神','正片还没上映，墓志铭先写好了'),
    ('贷款吹与反串','GENERAL','刚面试完，我感觉这工作稳了','工牌先定制一个'),
    ('贷款吹与反串','GENERAL','今天才背了两页书，感觉考试稳了','状元开始选庆功宴了'),
    ('贷款吹与反串','GENERAL','这商品还没发售就说销量世界第一','先把收据打印出来是吧'),
    ('贷款吹与反串','GENERAL','我刚练一天就感觉自己要无敌了','天才体验卡今日生效'),
    ('贷款吹与反串','GENERAL','这新手机还没发布就已经吊打一切了','参数没出，墓碑先立上了'),
    ('贷款吹与反串','GENERAL','这人一场没打已经被吹成第一了','世一体验卡能不能先试用'),
    ('回旋镖与清算','GENERAL','昨天还说他无敌，今天就说他该退了','评价的保质期只有一晚'),
    ('回旋镖与清算','GENERAL','之前说这游戏没人玩，今天自己又上线了','嘴先退游，手不同意'),
    ('回旋镖与清算','GENERAL','昨天嘲笑别人迟到，今天他自己睡过头','回旋镖精准制导'),
    ('回旋镖与清算','GENERAL','昨天说再也不点外卖，今天又点了','外卖软件的忠实反对派'),
    ('回旋镖与清算','GENERAL','开赛前说乱杀，打完开始找借口','赛前冠军，赛后天气预报员'),
    ('回旋镖与清算','GENERAL','他说自己绝不熬夜，结果三点还在线','生物钟第一个不服'),
    ('回旋镖与清算','GENERAL','昨天骂这个梗烂，今天他用得最欢','嘴上抵制，身体诚实'),
    ('回旋镖与清算','GENERAL','之前说这人不行，刚赢又开始吹了','立场跟着比分实时更新'),
    ('抽象短梗','GENERAL','有人开局就说这次必成','先质疑，再质疑'),
    ('抽象短梗','GENERAL','刚刚还挺安静的，一聊这事群就炸了','一键唤醒全群'),
    ('抽象短梗','GENERAL','这人每次都能找到借口','理由库存比装备栏还满'),
    ('抽象短梗','GENERAL','群里又开始无限复读了','复读机都得给你们让座'),
    ('抽象短梗','GENERAL','刚说一句就被截图了','互联网记忆今天没失灵'),
    ('抽象短梗','GENERAL','他发完这句就消失了','发言完毕，紧急避险'),
    ('抽象短梗','GENERAL','这都能被你预测对','蒙对一回直接封神'),
    ('抽象短梗','GENERAL','你们怎么什么事情都能吵起来','主打一个全民辩论赛'),
    ('粉丝争论与拆台','GENERAL','赢了就全是实力，输了就是运气不好','运气这口锅挺耐用'),
    ('粉丝争论与拆台','GENERAL','怎么别人犯错就挨骂，他犯错就叫可惜','双标开得比游戏还流畅'),
    ('粉丝争论与拆台','GENERAL','他一说偶像名字就什么都听不进去','耳朵自动开启粉丝过滤器'),
    ('粉丝争论与拆台','GENERAL','你们是不是谁红就吹谁','热度榜兼职实力榜'),
    ('粉丝争论与拆台','GENERAL','我觉得谁都不如我喜欢的那个','你这评价体系挺省事'),
    ('粉丝争论与拆台','GENERAL','他又开始拿十年前的荣誉压人了','历史成绩单都快翻烂了'),
    ('粉丝争论与拆台','GENERAL','一赢就说含金量高，一输就说没必要','含金量随结果自动调节'),
    ('粉丝争论与拆台','GENERAL','他刚说不在乎输赢就开始急了','不在乎三个字先红温了'),
    ('日常水帖接话','GENERAL','谁把闹钟关了我又迟到了','闹钟没输，是你不接招'),
    ('日常水帖接话','GENERAL','我又花半小时选外卖','外卖还没点，天都快黑了'),
    ('日常水帖接话','GENERAL','今天下班了我什么都不想干','下班后的主线任务是躺着'),
    ('日常水帖接话','GENERAL','刚打开书我就困了','知识的催眠效果稳定发挥'),
    ('日常水帖接话','GENERAL','我说只刷五分钟视频的','五分钟是按平台时间算的'),
    ('日常水帖接话','GENERAL','群里怎么突然没一个人说话','都在潜水赛区打决赛'),
    ('日常水帖接话','GENERAL','我电脑又蓝屏了','它先你一步申请下班'),
    ('日常水帖接话','GENERAL','今天又把奶茶洒了','奶茶先完成了自由落体'),
    ('赛事梗与讨论','LOL','这团战辅助冲进去五个人都没跟','你这开团属于独立团建'),
    ('赛事梗与讨论','LOL','我中单被对面压了六十刀','你是对面中单的经济顾问'),
    ('赛事梗与讨论','LOL','我打野整局没刷到一组野怪','打野位体验卡是观光用的'),
    ('赛事梗与讨论','LOL','说好的宇宙战舰怎么一轮游','战舰改潜艇了'),
    ('赛事梗与讨论','LOL','这把赢了就开始选冠军皮肤','皮肤策划已经开始加班了'),
    ('赛事梗与讨论','LOL','昨天还吹这队伍，今天就说解散','一晚上完成建队和清算'),
    ('事实与边界','GENERAL','听说有家公司要被收购了真的假的','先看公告，别把传言当新闻'),
    ('事实与边界','GENERAL','网上这个抗吧梗出自哪条帖子','得找原帖，不然容易串出处'),
    ('事实与边界','GENERAL','这条新闻是不是刚出来的','先看原消息，别把二传当官宣'),
    ('事实与边界','GENERAL','那个比赛到底几比几','哪场？说下队名我再核对'),
    ('事实与边界','GENERAL','你是不是知道内部消息','我可没内幕，等公开消息'),
    ('认真回应','GENERAL','今天心情很差，真的不想说话','那就歇会儿，今天不整活'),
    ('认真回应','GENERAL','我妈做手术了，最近事情特别多','先照顾好家里，别硬撑'),
    ('认真回应','GENERAL','我最近压力很大睡不好','先休息，别拿自己硬抗'),
    ('认真回应','GENERAL','最近家里有点事，没精力玩','先忙家里的事，游戏不急'),
]
HOLDOUT = [
    '我上单0/8，这把全怪打野', 'BLG今年绝对夺冠',
    '这选手赛前吹上天，赛后直接隐身', '我LOL已经十连败了',
    '今天不想打游戏了，感觉好累', '你们抗抗就知道串',
    '昨天说绝不熬夜，今天又通宵了', '我宣布这个新队已经是冠军了',
    '谁一开口群里就吵起来了', '昨天还看不起，今天怎么开始吹了',
    '我真的好累，不想说话', '这个梗最早是从哪来的',
]


def norm(v: str) -> str:
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', str(v or ''))).strip()


def key(v: str) -> str:
    return re.sub(r'[^\w\u4e00-\u9fff]', '', norm(v).casefold())


def digest(v: str) -> str:
    return hashlib.sha1(v.encode('utf-8')).hexdigest()[:20]


def csv_safe(v: str) -> str:
    v = str(v)
    return "'" + v if v.lstrip().startswith(('=', '+', '-', '@', '\t', '\r')) else v


def csv_restore(v: str) -> str:
    return v[1:] if v.startswith("'") and len(v) > 1 and v[1] in '=+-@\t\r' else v


def check_risk(p: str, c: str) -> str:
    blob = p + '\n' + c
    if not p or not c:
        return 'empty'
    if AD.search(blob) or BAD_LINK.search(blob):
        return 'ad_or_external_link'
    if PERSONAL.search(blob):
        return 'possible_private_contact'
    if ABUSE.search(blob):
        return 'severe_abuse'
    if UNVERIFIED_FACT.search(c):
        return 'unsupported_current_fact_in_reply'
    if UNVERIFIED_FACT.search(p) and PROMPT_QUESTION.search(p):
        return 'time_sensitive_factual_question'
    if EMOTION.search(p):
        return 'sensitive_emotion_needs_curated_reply'
    return ''


def game_of(text: str) -> str:
    v = bool(VAL.search(text))
    l = bool(LOL.search(text))
    if v and l:
        return 'MIXED'
    if v:
        return 'VALORANT'
    if l:
        return 'LOL'
    return 'GENERAL'


def anchors(text: str) -> set[str]:
    lo = text.casefold()
    out = set()
    for w in GAME_ANCHORS + KB_ANCHORS:
        # 单字（吹、串、典等）共现太容易误判上下文，不参与语义锚点。
        if len(w) < 2:
            continue
        if w.isascii():
            if re.search(r'(?<![a-z0-9])' + re.escape(w) + r'(?![a-z0-9])', lo):
                out.add(w)
        elif w in lo:
            out.add(w)
    return out


def bigrams(text: str) -> set[str]:
    cs = re.findall(r'[\u4e00-\u9fff]{2,}', text)
    out = set()
    for token in cs:
        out.update(token[i:i + 2] for i in range(len(token) - 1))
    return out - STOP_BI


def classify(p: str, c: str, g: str) -> str:
    """按抗吧互动形式分类，游戏标签只是元信息。"""
    blob = p + ' ' + c
    if re.search(r'好累|难受|压力大|生病|手术|住院|崩溃|很烦', p):
        return '认真回应'
    if re.search(r'出处|原帖|真的假的|官宣|新闻|哪年|谁是|几比几|真相|爆料', p):
        return '事实与边界'
    if re.search(r'回旋镖|打脸|清算|昨天.*今天|以前.*现在|之前.*现在|赛前.*赛后|输了.*吹|赢了.*骂', blob):
        return '回旋镖与清算'
    if re.search(r'夺冠|冠军|封神|第一|世一|稳了|无敌|预订|贷款|开香槟|宇宙战舰|吊打', p):
        return '贷款吹与反串'
    if re.search(r'粉丝|饭圈|黑粉|双标|孝|洗白|吹|喷|偶像|含金量', blob):
        return '粉丝争论与拆台'
    if re.search(r'选手|战队|比赛|团战|打野|大龙|赛区|季后赛|msi|lpl|lck|faker|blg|jdg|tes', blob, re.I):
        return '赛事梗与讨论'
    if REPLY_MEME.search(c) or STYLE.search(c):
        return '抽象短梗'
    return '日常水帖接话'


def score_pair(p: str, c: str) -> tuple[int, str, str, str, str]:
    p, c = norm(p), norm(c)
    risk = check_risk(p, c)
    if risk:
        return -99, '', '', '', risk
    if len(p) < 7 or len(p) > 190:
        return -99, '', '', '', 'prompt_length_outside_range'
    if len(c) < 3 or len(c) > 46:
        return -99, '', '', '', 'reply_length_outside_range'
    if key(p) == key(c):
        return -99, '', '', '', 'parroting_entire_prompt'
    if len(set(c)) <= 2 and len(c) > 8:
        return -99, '', '', '', 'character_repetition'
    if len(re.findall(r'[\u4e00-\u9fff]', c)) < 3:
        return -99, '', '', '', 'insufficient_chinese'
    g = game_of(p)
    reply_g = game_of(c)
    if g in ('LOL','VALORANT') and reply_g in ('LOL','VALORANT') and g != reply_g:
        return -99, '', '', '', 'game_mismatch'
    # 不因同时出现LOL/其他游戏内容就一律淘汰：抗吧不是某一个游戏板块。
    common = anchors(p) & anchors(c)
    shared_grams = bigrams(p) & bigrams(c)
    shared_grams = {x for x in shared_grams if x not in {'还是','不是','可能','一样','知道','这个','哪个','怎么','这个'}}
    has_style = bool(STYLE.search(c))
    score = 0
    reasons = []
    if len(c) <= 24:
        score += 4
        reasons.append('short_4')
    elif len(c) <= 34:
        score += 3
        reasons.append('short_3')
    else:
        score += 1
        reasons.append('longer_reply')
    if len(p) <= 95:
        score += 1
    if common:
        score += 4
        reasons.append('shared_anchor:' + '/'.join(sorted(common)[:3]))
    elif len(shared_grams) >= 2:
        score += 2
        reasons.append('shared_bigrams')
    elif shared_grams:
        score += 1
        reasons.append('weak_bigrams')
    else:
        reasons.append('NO_CONTEXT_MATCH_REVIEW')
    if has_style:
        score += 2
        reasons.append('meme_or_reversal_style')
    if c.endswith(('是吧', '？', '?')):
        score += 1
        reasons.append('short_comeback')
    if REPLY_MEME.search(c):
        score += 2
        reasons.append('kb_conversational_style')
    if g == 'GENERAL' and reply_g in ('LOL','VALORANT'):
        score -= 2
        reasons.append('unprompted_game_switch')
    # 自动候选至少应有文本锚点，避免随机楼层回复靠风格关键词混进训练集。
    if not common and len(shared_grams) < 2:
        return score, '', g, ';'.join(reasons), 'weak_context_needs_manual_rewrite'
    return score, classify(p, c, g), g, ';'.join(reasons), ''


def back_up(path: Path):
    if path.is_file():
        dest = DATA / 'v4_kangba_backups' / datetime.now().strftime('%Y%m%d_%H%M%S')
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest / path.name)


def write_csv(path: Path, rows: list[dict]):
    DATA.mkdir(parents=True, exist_ok=True)
    back_up(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction='ignore')
        w.writeheader()
        for row in rows:
            w.writerow({k: csv_safe(row.get(k, '')) for k in FIELDS})
    tmp.replace(path)


def load_review_csv(path: Path):
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        for row in csv.DictReader(f):
            yield {k: csv_restore(v or '') for k, v in row.items()}


def old_decisions(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    return {row['id']: row['approve'].strip().lower() for row in load_review_csv(path)
            if row.get('approve', '').strip().lower() in ('y','n')}


def source_index() -> dict[str, tuple[str, str]]:
    if not REVIEW_SOURCE.is_file():
        return {}
    mapping = {}
    with REVIEW_SOURCE.open('r', encoding='utf-8-sig', newline='') as f:
        for item in csv.DictReader(f):
            p, c = norm(csv_restore(item.get('prompt', ''))), norm(csv_restore(item.get('completion', '')))
            if p and c:
                mapping.setdefault(digest(key(p) + '\n' + key(c)),
                                   (item.get('post_id', ''), item.get('source_url', '')))
    return mapping


def load_source(path: Path):
    if not path.is_file():
        raise SystemExit('缺少训练源文件：' + str(path) + '\n请确认 QWbot/datasets/kangba_train.jsonl 存在')
    with path.open('r', encoding='utf-8') as f:
        for idx, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                prompt_msgs, completion = item['prompt'], item['completion']
                prompt = '\n'.join(m['content'] for m in prompt_msgs if m['role'] == 'user')
                answer = completion[-1]['content']
                if isinstance(prompt, str) and isinstance(answer, str):
                    yield idx, norm(prompt), norm(answer)
            except (KeyError, ValueError, TypeError, IndexError):
                continue


def seed_examples(old: dict[str,str]) -> list[dict]:
    rows = []
    for i, (cat, game, p, c) in enumerate(SEED_EXAMPLES, 1):
        identifier = 'seed_' + digest(p + '\n' + c)
        rows.append(dict(approve=old.get(identifier, 'y'), id=identifier, score=99,
                         category=cat, game=game, context_signals='人工编写的QQ场景示例',
                         review_warning='原创示例，非贴吧原文', origin='authored_example',
                         post_id=f'authored_{i}', source_url='', prompt=p, completion=c))
    return rows


def cmd_extract(args):
    old = old_decisions(CANDIDATES)
    old_seeds = old_decisions(SEEDS)
    idx = source_index()
    counter = Counter()
    provisional = []
    holdout_keys = {key(x) for x in HOLDOUT}
    for line, prompt, completion in load_source(SOURCE):
        counter['read_rows'] += 1
        s, cat, game, reasons, rej = score_pair(prompt, completion)
        if rej:
            counter['reject_' + rej] += 1
            continue
        if s < args.min_score:
            counter['below_min_score'] += 1
            continue
        if key(prompt) in holdout_keys:
            counter['holdout_held_back'] += 1
            continue
        pair_key = digest(key(prompt) + '\n' + key(completion))
        post_id, url = idx.get(pair_key, ('', ''))
        rid = 'forum_' + pair_key
        provisional.append(dict(approve=old.get(rid, ''), id=rid, score=s,
                                category=cat, game=game, context_signals=reasons,
                                review_warning='未人工审核；词面相关不代表楼层真实接话；赛事或人物内容须核实', origin='forum_candidate',
                                post_id=post_id or ('prompt_' + digest(key(prompt))),
                                source_url=url, prompt=prompt, completion=completion))
        counter['scored_candidates'] += 1
    provisional.sort(key=lambda r: (-r['score'], r['id']))
    counts = Counter()
    unique_pairs, completion_counts, prompt_counts, posts = set(), Counter(), Counter(), Counter()
    by_category = defaultdict(list)
    for row in provisional:
        pkey, ckey = key(row['prompt']), key(row['completion'])
        if (pkey, ckey) in unique_pairs:
            counter['dedup_pair'] += 1
            continue
        if completion_counts[ckey] >= args.max_per_reply or prompt_counts[pkey] >= args.max_per_prompt:
            counter['frequency_cap'] += 1
            continue
        if posts[row['post_id']] >= args.max_per_post:
            counter['post_cap'] += 1
            continue
        unique_pairs.add((pkey, ckey))
        completion_counts[ckey] += 1
        prompt_counts[pkey] += 1
        posts[row['post_id']] += 1
        by_category[row['category']].append(row)
    # 最大限度平衡场景；数据中缺失的场景不伪造真实贴吧回复。
    quotas = {
        '贷款吹与反串': .18, '回旋镖与清算': .19, '抽象短梗': .23,
        '粉丝争论与拆台': .13, '日常水帖接话': .17, '赛事梗与讨论': .07,
        '事实与边界': .02, '认真回应': .01,
    }
    selected = []
    selected_ids = set()
    for category, fraction in quotas.items():
        n = round(args.limit * fraction)
        for row in by_category[category][:n]:
            selected.append(row)
            selected_ids.add(row['id'])
    rest = sorted((r for lst in by_category.values() for r in lst if r['id'] not in selected_ids),
                  key=lambda r: (-r['score'], r['id']))
    selected.extend(rest[:max(0, args.limit - len(selected))])
    selected.sort(key=lambda r: (-r['score'], r['category'], r['id']))
    for r in selected:
        counts[r['category']] += 1
    seeds = seed_examples(old_seeds)
    write_csv(CANDIDATES, selected)
    write_csv(SEEDS, seeds)
    EVAL_PROMPTS.write_text('\n'.join(HOLDOUT) + '\n', encoding='utf-8')
    report = dict(stage='extract', original_count=counter['read_rows'],
                  candidates=len(selected), authored_seed_count=len(seeds),
                  categories=dict(counts), diagnostics=dict(counter),
                  note='不限LOL话题。论坛自动筛选只产生未审核候选，不代表语义或事实核验；原文件不改。原创示范独立标明。')
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print('读取原训练集:', counter['read_rows'])
    print('V4待审核论坛候选:', len(selected), '| 原创示例:', len(seeds))
    print('场景分布:', dict(counts))
    print('已生成:', CANDIDATES, '\n         ', SEEDS, '\n         ', REPORT)
    print('下一步：审核候选 approve 列填 y，或用 build --auto-top 250 做未经审核的流程实验（不推荐正式微调）。')


def groupkey(row: dict) -> str:
    # 同帖子尽可能在同侧；没有来源帖ID时用相同prompt分组。
    value = row.get('post_id', '')
    if value.startswith('authored_') or not value:
        value = 'p_' + digest(key(row['prompt']))
    return value


def cmd_build(args):
    if not CANDIDATES.exists() or not SEEDS.exists():
        raise SystemExit('请先执行 python prepare_kangba_v4_kangba.py extract')
    candidates = list(load_review_csv(CANDIDATES))
    seeds = list(load_review_csv(SEEDS))
    accepted = []
    seen_ids = set()
    stats = Counter()
    for row in seeds + candidates:
        approval = row.get('approve', '').strip().lower()
        if approval in ('n', 'no', '0'):
            stats['manual_rejected'] += 1
            continue
        if approval == 'y':
            accepted.append(row)
            seen_ids.add(row['id'])
            stats['approved'] += 1
    if args.auto_top > 0:
        remaining = [r for r in candidates if r['id'] not in seen_ids and r.get('approve','').lower() not in ('n','no','0')]
        remaining.sort(key=lambda r: (-int(r.get('score',0)), r['id']))
        for row in remaining[:args.auto_top]:
            row = dict(row)
            row['origin'] = 'forum_UNREVIEWED_autoexperiment'
            accepted.append(row)
            stats['unreviewed_auto_added'] += 1
    # 第二次风控：防止CSV审核时意外将危险文本重新放进模型。
    filtered = []
    used_pairs, answer_counts, prompt_counts = set(), Counter(), Counter()
    holdout_keys = {key(x) for x in HOLDOUT}
    for row in accepted:
        p, c = norm(row.get('prompt', '')), norm(row.get('completion', ''))
        risk = check_risk(p, c)
        if risk and row.get('origin') != 'authored_example':
            stats['rejected_on_build_' + risk] += 1
            continue
        if not p or not c or key(p) == key(c) or key(p) in holdout_keys:
            stats['invalid_or_holdout'] += 1
            continue
        pkey, ckey = key(p), key(c)
        if (pkey, ckey) in used_pairs:
            stats['dup_pair'] += 1
            continue
        if answer_counts[ckey] >= args.max_per_reply or prompt_counts[pkey] >= args.max_per_prompt:
            stats['build_frequency_cap'] += 1
            continue
        used_pairs.add((pkey, ckey))
        answer_counts[ckey] += 1
        prompt_counts[pkey] += 1
        filtered.append({**row,'prompt':p,'completion':c})
    if len(filtered) < 20:
        raise SystemExit(f'仅筛得 {len(filtered)} 条，至少需要20条，请先审核或添加 --auto-top 250')
    grouping = defaultdict(list)
    for row in filtered:
        grouping[groupkey(row)].append(row)
    rng = random.Random(args.seed)
    keys = sorted(grouping)
    rng.shuffle(keys)
    n_valid = max(1, min(len(keys) - 1, round(len(keys) * args.valid_fraction)))
    valid_set = set(keys[:n_valid])
    train_rows, valid_rows = [], []
    for grp, batch in grouping.items():
        (valid_rows if grp in valid_set else train_rows).extend(batch)
    # 不让相同答案或相同问题同时出现在训练与验证中。
    train_prompts = {key(r['prompt']) for r in train_rows}
    train_answers = {key(r['completion']) for r in train_rows}
    clean_valid = [r for r in valid_rows if key(r['prompt']) not in train_prompts
                   and key(r['completion']) not in train_answers]
    stats['valid_removed_for_leakage'] = len(valid_rows) - len(clean_valid)
    valid_rows = clean_valid
    if not train_rows or not valid_rows:
        raise SystemExit('训练/验证划分为空；需更多不同帖子和回复，或调整 --valid-fraction')
    def item(r):
        return {'prompt':[{'role':'system','content':SYSTEM},{'role':'user','content':r['prompt']}],
                'completion':[{'role':'assistant','content':r['completion']}]}
    for path, rows in ((TRAIN,train_rows),(VALID,valid_rows)):
        back_up(path)
        temp = path.with_suffix(path.suffix + '.tmp')
        with temp.open('w',encoding='utf-8') as f:
            for r in rows:
                f.write(json.dumps(item(r),ensure_ascii=False) + '\n')
        temp.replace(path)
    meta_rows = train_rows + valid_rows
    metadata_header = FIELDS + ['split']
    back_up(META)
    with META.open('w',encoding='utf-8-sig',newline='') as f:
        w = csv.DictWriter(f,fieldnames=metadata_header,extrasaction='ignore')
        w.writeheader()
        train_ids = {r['id'] for r in train_rows}
        for row in meta_rows:
            record={**row,'split':'train' if row['id'] in train_ids else 'valid'}
            w.writerow({k:csv_safe(record.get(k,'')) for k in metadata_header})
    report = dict(stage='build', train=len(train_rows), valid=len(valid_rows),
                  trained_on_unreviewed_auto_count=sum(r['origin']=='forum_UNREVIEWED_autoexperiment' for r in meta_rows),
                  train_category_counts=dict(Counter(r['category'] for r in train_rows)),
                  valid_category_counts=dict(Counter(r['category'] for r in valid_rows)),
                  diagnostics=dict(stats),
                  note='JSONL是prompt/completion格式，与V3训练脚本格式相同；不能原样套用V3的训练输入路径。')
    REPORT.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('V4训练集:',len(train_rows),TRAIN)
    print('V4验证集:',len(valid_rows),VALID)
    print('未经审核自动实验样本:',report['trained_on_unreviewed_auto_count'])
    print('构建报告:',REPORT)


def cmd_stats(_):
    if not REPORT.exists():
        raise SystemExit('尚未生成报告；先运行 extract')
    print(REPORT.read_text(encoding='utf-8'))


def main():
    p = argparse.ArgumentParser(description='抗压背锅吧V4：不限话题、以反串接话风格筛选候选、质量审核、SFT数据集')
    sub = p.add_subparsers(dest='command',required=True)
    x=sub.add_parser('extract',help='从旧kangba_train.jsonl提取约1000条抗吧风格待审核样本')
    x.add_argument('--limit',type=int,default=1000)
    x.add_argument('--min-score',type=int,default=7)
    x.add_argument('--max-per-reply',type=int,default=1)
    x.add_argument('--max-per-prompt',type=int,default=2)
    x.add_argument('--max-per-post',type=int,default=3)
    x.set_defaults(func=cmd_extract)
    x=sub.add_parser('build',help='人工选中样本+原创示例 -> 训练/验证JSONL')
    x.add_argument('--auto-top',type=int,default=0,help='额外纳入N条未经审核高分候选，仅供实验；不是质量担保')
    x.add_argument('--valid-fraction',type=float,default=.12)
    x.add_argument('--seed',type=int,default=3407)
    x.add_argument('--max-per-reply',type=int,default=1)
    x.add_argument('--max-per-prompt',type=int,default=2)
    x.set_defaults(func=cmd_build)
    x=sub.add_parser('stats',help='查看最近一次筛选/构建报告')
    x.set_defaults(func=cmd_stats)
    args=p.parse_args()
    if hasattr(args,'limit') and args.limit < 1:
        p.error('--limit 必须 >=1')
    if hasattr(args,'valid_fraction') and not (0 < args.valid_fraction < 1):
        p.error('--valid-fraction 必须在0到1之间')
    args.func(args)

if __name__=='__main__':
    main()
