#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""抗压背锅吧语料 V3 · 高保留率版（Python 标准库）

目标：保留原始贴吧语料中的真实回复，不因短、长、重复、非电竞、低信息量或帖子配额而删除；
     仅剔除明确广告/引流、空回复，针对敏感联系方式先脱敏；明显人身安全威胁默认不进入训练。

原始数据：datasets/kangba_raw/data/**/tieba_post_detail_page_*.json
运行：
  python prepare_kangba_corpus.py extract
  python prepare_kangba_corpus.py build
  python prepare_kangba_corpus.py stats

新增：
  datasets/kangba_review.csv             全量保留样本（默认 approve=y，可手动标记 n）
  datasets/kangba_recovered_samples.csv  过去会被短句/长度/非电竞/重复规则过滤的已保留样本
  datasets/kangba_short_memes.csv        短回复出现频率参考表（不单独重复加入训练）
  datasets/kangba_rejected_samples.csv  真正剔除的内容及原因（广告只保留掩码，不扩散联系方式）
  datasets/kangba_extraction_stats.json  统计信息
  datasets/kangba_train.jsonl           训练样本
  datasets/kangba_valid.jsonl           验证样本

注意：原始回复未必回复楼主，全部保留并不代表全部适合 SFT。请审查许可、个人信息与实际问答质量。
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

csv.field_size_limit(10_000_000)

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / 'datasets'
SOURCE_DIR = DATA_DIR / 'kangba_raw' / 'data'
REVIEW_FILE = DATA_DIR / 'kangba_review.csv'
RECOVERED_FILE = DATA_DIR / 'kangba_recovered_samples.csv'
REJECT_FILE = DATA_DIR / 'kangba_rejected_samples.csv'
SHORT_FILE = DATA_DIR / 'kangba_short_memes.csv'
STATS_FILE = DATA_DIR / 'kangba_extraction_stats.json'
TRAIN_FILE = DATA_DIR / 'kangba_train.jsonl'
VALID_FILE = DATA_DIR / 'kangba_valid.jsonl'

SYSTEM = (
    '你是QQ群里的AI群友，擅长LOL、瓦洛兰特和抗压背锅吧式反串。'
    '普通闲聊用自然短句，不写客服腔；需要查证的真实赛事和人物信息不编造。'
)

# 只有明确引流/促销话术才认作广告；提及“京东”“淘宝”本身不应被删除。
AD_STRONG = re.compile(
    r'购物礼金|优惠礼包|领红包|返现(?:活动|领取)|刷单|兼职日结|推广邀请码|'
    r'扫码(?:进群|购买|领取|加群)|加(?:我)?(?:微|v|vx)(?:信)?(?:领|买|拿|咨询|下单)|'
    r'代练接单|陪玩接单|招商加盟|招代理|买粉|出售账号|联系(?:我)?(?:微信|QQ|vx)|'
    r'京东首页搜|淘宝(?:店铺|优惠券)|拼多多(?:砍价|领券)|'
    r'点击(?:链接|购买)|联系(?:客服|商家)下单|优惠券(?:领取|入口)',
    re.I,
)
AD_LINK = re.compile(r'(?:https?://|www\.|t\.cn/|bit\.ly/|短链|vx[:：]|微信[:：]|QQ群[:：])', re.I)
AD_CONTEXT = re.compile(r'购买|优惠|下单|私聊|推广|扫码|返现|免费领|加群|客服|福利|赚米|包邮|直降|出售', re.I)
STRONG_THREAT = re.compile(r'杀你全家|死全家|人肉搜索(?:地址|电话)|开盒(?:地址|家人)|种族灭绝', re.I)

# 确认或近似个人联系方式只做脱敏，保留其余自然对话。
PHONE = re.compile(r'(?<!\d)1[3-9]\d{9}(?!\d)')
EMAIL = re.compile(r'(?<![\w.-])[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}(?![\w.-])')
CHAT_ID = re.compile(r'(?:(?:微信|vx|V信|QQ号|QQ号码)\s*[:：=]\s*)[a-zA-Z0-9_-]{5,}', re.I)

# 仅用于“以前为什么可能被排除”的可审计标签，不参与剔除。
GAME_RE = re.compile(
    r'英雄联盟|\blol\b|\blpl\b|\blck\b|\bmsi\b|世界赛|上单|打野|中单|辅助|'
    r'无畏契约|瓦洛兰特|\bvalorant\b|\bvct\b|排位|峡谷|\bedg\b|\bblg\b|\bt1\b|'
    r'faker|uzi|theshy|康康|zmjjkk|smoggy|电竞|团战|补枪|枪法', re.I
)
LOW_INFO = re.compile(r'^(?:[哈啊呃哦嗯啧哎]{3,}|666+|\d+|顶|插眼|蹲|前排|cy|dd|mark|确实)$', re.I)
SHORT_NAMES = {'典', '绷', '乐', '急', '急了', '难绷', '闹麻了', '伟大', '幽默', '逆天', '纯路人',
               '红温', '下饭', '坐牢', '世一', '就这', '好好好', '清算', '贷款', '破防', '太典了'}

REVIEW_FIELDS = ['approve', 'id', 'post_id', 'reply_index', 'recovery_reason', 'corpus_freq',
                 'title', 'body_excerpt', 'prompt', 'completion', 'source_url']
RECOVERED_FIELDS = REVIEW_FIELDS
REJECT_FIELDS = ['reason', 'post_id', 'reply_index', 'title', 'reply_preview', 'source_url']
SHORT_FIELDS = ['meme', 'corpus_freq', 'post_freq', 'example_prompt', 'example_source_url']


def clean(value: object) -> str:
    if not isinstance(value, str):
        return ''
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', value)).strip()


def redact(text: str) -> str:
    text = EMAIL.sub('[邮箱已脱敏]', text)
    text = PHONE.sub('[手机号已脱敏]', text)
    text = CHAT_ID.sub('[联系方式已脱敏]', text)
    return text


def safe_for_csv(text: object) -> object:
    """防止 WPS/Excel 将论坛文本当作表格公式执行。"""
    if isinstance(text, str) and text.lstrip().startswith(('=', '+', '-', '@', '\t', '\r')):
        return "'" + text
    return text


def restore_csv(text: str) -> str:
    return text[1:] if text.startswith("'") and len(text) > 1 and text[1] in '=+-@\t\r' else text


def is_ad(text: str) -> bool:
    if AD_STRONG.search(text):
        return True
    return bool(AD_LINK.search(text) and AD_CONTEXT.search(text))


def reason_for_exclusion(text: str) -> str | None:
    if not text:
        return 'empty'
    if is_ad(text):
        return 'advertisement'
    if STRONG_THREAT.search(text):
        return 'explicit_threat'
    return None


def legacy_reasons(text: str, topic: str, seen: set[str]) -> list[str]:
    """模拟旧版常见淘汰缘由，用于恢复样本的标注，而不是过滤条件。"""
    result = []
    if not GAME_RE.search(topic):
        result.append('formerly_non_gaming')
    if not 6 <= len(text) <= 75:
        result.append('formerly_bad_length')
    if LOW_INFO.fullmatch(text) or len(re.sub(r'[\W_]+', '', text)) < 5:
        result.append('formerly_low_content')
    key = re.sub(r'[\W_]+', '', text.casefold())
    if key in seen:
        result.append('formerly_duplicate')
    seen.add(key)
    return result


def load_approvals() -> tuple[dict[str, str], dict[str, str]]:
    """加载旧版人工审批，保留明确否决；旧版空白在新版默认启用。"""
    by_id, by_legacy = {}, {}
    if not REVIEW_FILE.exists():
        return by_id, by_legacy
    with REVIEW_FILE.open('r', encoding='utf-8-sig', newline='') as f:
        for row in csv.DictReader(f):
            status = (row.get('approve') or '').strip().lower()
            rid = row.get('id', '')
            if rid:
                by_id[rid] = status
            if row.get('post_id') and row.get('completion'):
                legacy = hashlib.sha1((row['post_id'] + '\0' + clean(restore_csv(row['completion']))).encode()).hexdigest()[:20]
                by_legacy[legacy] = status
    return by_id, by_legacy


def archive_existing(paths: list[Path]) -> None:
    present = [p for p in paths if p.exists()]
    if not present:
        return
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backdir = DATA_DIR / 'backups' / ('before_v3_' + stamp)
    backdir.mkdir(parents=True, exist_ok=True)
    for p in present:
        shutil.copy2(p, backdir / p.name)
    print('已备份旧结果:', backdir)


def writer(path: Path, fields: list[str]):
    f = path.open('w', encoding='utf-8-sig', newline='')
    w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
    w.writeheader()
    return f, w


def write_row(w: csv.DictWriter, row: dict) -> None:
    w.writerow({name: safe_for_csv(row.get(name, '')) for name in w.fieldnames})


def fill_frequency_column(csv_tmp: Path, fields: list[str], freq: Counter) -> None:
    """第二遍回填真实语料出现频率，避免输出虚假的0。"""
    staging = csv_tmp.with_name(csv_tmp.name + '.freq')
    with csv_tmp.open('r', encoding='utf-8-sig', newline='') as src, \
         staging.open('w', encoding='utf-8-sig', newline='') as dest:
        reader = csv.DictReader(src)
        writer_ = csv.DictWriter(dest, fieldnames=fields, extrasaction='ignore')
        writer_.writeheader()
        for row in reader:
            text = restore_csv(row.get('completion', ''))
            key = re.sub(r'[\W_]+', '', text.casefold())
            row['corpus_freq'] = freq[key]
            write_row(writer_, row)
    staging.replace(csv_tmp)


def cmd_extract(_: argparse.Namespace) -> None:
    paths = sorted(SOURCE_DIR.rglob('tieba_post_detail_page_*.json'))
    if not paths:
        raise SystemExit(f'原始JSON不存在：{SOURCE_DIR}')
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    approvals, legacy_approvals = load_approvals()
    archive_existing([REVIEW_FILE, RECOVERED_FILE, REJECT_FILE, SHORT_FILE, STATS_FILE])
    files = []
    stats = Counter()
    seen = set()
    freq = Counter()
    short_freq = Counter()
    short_posts = defaultdict(set)
    short_example = {}
    # 写到 .tmp，整个提取成功后再覆盖当前文件。
    paths_out = [(REVIEW_FILE, REVIEW_FIELDS), (RECOVERED_FILE, RECOVERED_FIELDS), (REJECT_FILE, REJECT_FIELDS)]
    try:
        holders = []
        for out, fields in paths_out:
            f, w = writer(out.with_name(out.name + '.tmp'), fields)
            holders.append((f, w))
        (all_f, all_w), (rec_f, rec_w), (rej_f, rej_w) = holders
        files = [all_f, rec_f, rej_f]
        for path in paths:
            try:
                posts = json.loads(path.read_text(encoding='utf-8'))
            except (OSError, UnicodeError, json.JSONDecodeError) as e:
                stats['file_read_errors'] += 1
                print('[WARN] 读取失败:', path, str(e)[:90])
                continue
            if not isinstance(posts, list):
                stats['invalid_json'] += 1
                continue
            stats['files'] += 1
            for ix, item in enumerate(posts):
                stats['posts'] += 1
                if not isinstance(item, dict):
                    stats['invalid_post'] += 1
                    continue
                raw_title = clean(item.get('标题'))
                raw_body = clean(item.get('楼主内容'))
                title, body = redact(raw_title), redact(raw_body)
                url = clean(item.get('链接'))
                post_id = hashlib.sha1(f'{path.name}:{ix}:{url}'.encode()).hexdigest()[:16]
                raw_replies = item.get('回复列表', [])
                if not isinstance(raw_replies, list):
                    stats['invalid_replies'] += 1
                    continue
                # 即使标题/主楼包含广告，也只删除广告段落；不整帖剔除普通回复。
                prompt = title
                if body and body != title and not is_ad(body) and not STRONG_THREAT.search(body):
                    prompt = (title + '\n' + body) if title else body
                if not prompt or is_ad(prompt) or STRONG_THREAT.search(prompt):
                    prompt = '[贴吧讨论帖]'
                    stats['prompt_redacted'] += 1
                prompt = prompt[:1000]  # 仅限制训练输入上下文；原帖完整正文仍保存在原始JSON
                topic = raw_title + ' ' + raw_body[:260]
                for j, original in enumerate(raw_replies):
                    stats['replies_seen'] += 1
                    t = clean(original)
                    excluded = reason_for_exclusion(t)
                    if excluded:
                        stats['removed_' + excluded] += 1
                        preview = '[广告内容已隐藏]' if excluded == 'advertisement' else ('[违规威胁内容已隐藏]' if excluded == 'explicit_threat' else '')
                        write_row(rej_w, {'reason': excluded, 'post_id': post_id,
                                          'reply_index': j, 'title': title[:120],
                                          'reply_preview': preview, 'source_url': url})
                        continue
                    reply = redact(t)
                    oldreason = legacy_reasons(reply, topic, seen)
                    recovery = ','.join(oldreason)
                    uid = hashlib.sha1(f'{post_id}\0{j}\0{reply}'.encode()).hexdigest()[:20]
                    legacy_id = hashlib.sha1(f'{post_id}\0{reply}'.encode()).hexdigest()[:20]
                    explicit = approvals.get(uid, legacy_approvals.get(legacy_id, ''))
                    # V3“全部启用”：不经人工审核的回复也先批准；明确否决的继续排除。
                    approve = 'n' if explicit in {'n', 'no', '0', '否', '拒绝'} else 'y'
                    row = {'approve': approve, 'id': uid, 'post_id': post_id, 'reply_index': j,
                           'recovery_reason': recovery, 'corpus_freq': 0,
                           'title': title[:400], 'body_excerpt': body[:400],
                           'prompt': prompt, 'completion': reply, 'source_url': url}
                    write_row(all_w, row)
                    stats['retained'] += 1
                    if recovery:
                        write_row(rec_w, row)
                        stats['recovered_rows'] += 1
                    key = re.sub(r'[\W_]+', '', reply.casefold())
                    freq[key] += 1
                    if len(reply) <= 10:
                        short_freq[reply] += 1
                        short_posts[reply].add(post_id)
                        short_example.setdefault(reply, (prompt, url))
        for f in files:
            f.close()
        for out, fields in paths_out[:2]:
            fill_frequency_column(out.with_name(out.name + '.tmp'), fields, freq)
        for out, _ in paths_out:
            out.with_name(out.name + '.tmp').replace(out)
    except BaseException:
        for f in files:
            f.close()
        raise
    # 主表/恢复表已回填真实 corpus_freq；短梗另汇总频率和跨帖子次数。
    with SHORT_FILE.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=SHORT_FIELDS)
        w.writeheader()
        for meme, count in sorted(short_freq.items(), key=lambda x: (-x[1], x[0])):
            prompt, url = short_example[meme]
            write_row(w, {'meme': meme, 'corpus_freq': count, 'post_freq': len(short_posts[meme]),
                          'example_prompt': prompt[:400], 'example_source_url': url})
    summary = {
        'counts': dict(stats), 'unique_reply_texts': len(freq),
        'notes': [
            '所有非广告、非空白、非明显严重威胁回复均保留；短句/重复/非游戏/低信息量不会被直接过滤。',
            '旧版过滤原因只用于 kangba_recovered_samples.csv 的审计标签，不代表原脚本完整重现。',
            '保留重复帖内回复；按出现次数分配训练权重，可能造成模型复读或偏见。',
            '回复可能回答的是其他楼层，不一定回答主楼；未经人工审查并不等于高质量。',
            '短梗出现次数仅代表本次采集的语料，不能证明全网热度。',
            '部分联系方式已脱敏，训练前仍应复核私密信息和数据许可。',
        ],
    }
    STATS_FILE.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print('JSON文件:', stats['files'], '| 帖子:', stats['posts'], '| 原始回复:', stats['replies_seen'])
    print('全部启用（含原低质量/重复/短句）:', stats['retained'])
    print('其中原规则可能排除但已恢复:', stats['recovered_rows'])
    print('剔除空白:', stats['removed_empty'], '| 广告:', stats['removed_advertisement'],
          '| 明显严重威胁:', stats['removed_explicit_threat'])
    print('CSV:', REVIEW_FILE, RECOVERED_FILE, REJECT_FILE, SHORT_FILE, sep='\n  ')
    print('训练集生成：python prepare_kangba_corpus.py build')


def cmd_build(args: argparse.Namespace) -> None:
    if not REVIEW_FILE.exists():
        raise SystemExit('请先执行 extract')
    if not 0 < args.valid_fraction < 1:
        raise SystemExit('--valid-fraction 必须在 0 到 1 之间')
    # 第一遍按帖子ID确定验证集，避免同一帖子跨训练与验证。
    post_ids = set()
    counts = Counter()
    with REVIEW_FILE.open('r', encoding='utf-8-sig', newline='') as f:
        for row in csv.DictReader(f):
            counts['rows'] += 1
            if (row.get('approve') or '').strip().lower() in {'n', 'no', '0', '否', '拒绝'}:
                counts['manual_rejected'] += 1
                continue
            post_ids.add(row['post_id'])
            counts['approved'] += 1
    if len(post_ids) < 2:
        raise SystemExit('至少需要两个帖子的有效训练数据')
    rng = random.Random(args.seed)
    ids = sorted(post_ids)
    rng.shuffle(ids)
    n_valid = max(1, min(len(ids) - 1, round(len(ids) * args.valid_fraction)))
    valid_ids = set(ids[:n_valid])
    archive_existing([TRAIN_FILE, VALID_FILE])
    with REVIEW_FILE.open('r', encoding='utf-8-sig', newline='') as src, \
         TRAIN_FILE.open('w', encoding='utf-8') as train_out, \
         VALID_FILE.open('w', encoding='utf-8') as valid_out:
        for row in csv.DictReader(src):
            if (row.get('approve') or '').strip().lower() in {'n', 'no', '0', '否', '拒绝'}:
                continue
            prompt = restore_csv((row.get('prompt') or '').strip())
            completion = restore_csv((row.get('completion') or '').strip())
            if not prompt or not completion:
                counts['skipped_empty'] += 1
                continue
            # 针对编辑后的CSV再次检查广告。
            if is_ad(prompt) or reason_for_exclusion(completion):
                counts['skipped_new_spam'] += 1
                continue
            message = {
                'prompt': [
                    {'role': 'system', 'content': SYSTEM},
                    {'role': 'user', 'content': prompt},
                ],
                'completion': [{'role': 'assistant', 'content': completion}],
            }
            out = valid_out if row['post_id'] in valid_ids else train_out
            out.write(json.dumps(message, ensure_ascii=False) + '\n')
            counts['valid' if row['post_id'] in valid_ids else 'train'] += 1
    print('CSV原始行数:', counts['rows'], '| 全部启用:', counts['approved'],
          '| 人工否决:', counts['manual_rejected'])
    print('训练:', counts['train'], TRAIN_FILE)
    print('验证:', counts['valid'], VALID_FILE)
    print('跳过:', counts['skipped_empty'] + counts['skipped_new_spam'])
    print('提醒：同一句重复回复会保留多次；未经审核的问答配对可能错误。')


def cmd_stats(_: argparse.Namespace) -> None:
    if not STATS_FILE.exists():
        raise SystemExit('尚无统计，先执行 extract')
    print(STATS_FILE.read_text(encoding='utf-8'))


def main() -> None:
    parser = argparse.ArgumentParser(description='抗吧语料V3：全量保留，主要删除广告，默认全部启用')
    sub = parser.add_subparsers(dest='command', required=True)
    ext = sub.add_parser('extract', help='重新扫描所有原始JSON，写全量审核表和恢复样本表')
    ext.set_defaults(func=cmd_extract)
    build = sub.add_parser('build', help='将全部启用的语料按帖子划分为训练/验证集')
    build.add_argument('--valid-fraction', type=float, default=0.1)
    build.add_argument('--seed', type=int, default=3407)
    build.add_argument('--include-short', action='store_true', help='兼容旧版命令；短回复已默认全部纳入，不再重复添加')
    build.set_defaults(func=cmd_build)
    stat = sub.add_parser('stats', help='打印上次提取的统计数据')
    stat.set_defaults(func=cmd_stats)
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
