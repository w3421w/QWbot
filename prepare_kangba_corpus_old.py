#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""抗吧语料清洗 V2：保留审核流程、记录被拒原因、统计短梗出现次数。

原始数据路径：datasets/kangba_raw/data/**/tieba_post_detail_page_*.json

用法：
  python prepare_kangba_corpus.py extract
  # 审核 datasets/kangba_review.csv 和 datasets/kangba_short_memes.csv 的 approve 列
  python prepare_kangba_corpus.py build
  python prepare_kangba_corpus.py build --include-short  # 可选：吸收已审核短梗，默认不吸收

注意：帖子回复可能是回复其他楼层而不是楼主；自动排序不等于正确配对。
训练前请核实语料使用授权、个人信息及样本质量。
只依赖 Python 标准库，Windows/WSL 均可运行。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import shutil
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "datasets"
SOURCE_DIR = DATA_DIR / "kangba_raw" / "data"
REVIEW_FILE = DATA_DIR / "kangba_review.csv"
REJECT_FILE = DATA_DIR / "kangba_rejected_samples.csv"
SHORT_FILE = DATA_DIR / "kangba_short_memes.csv"
STATS_FILE = DATA_DIR / "kangba_extraction_stats.json"
TRAIN_FILE = DATA_DIR / "kangba_train.jsonl"
VALID_FILE = DATA_DIR / "kangba_valid.jsonl"

SYSTEM = (
    "你是QQ群里的AI群友，偏抗压背锅吧串子风格。"
    "日常一两句，短、自然、会反串和接梗；不编造比赛、转会或真人传闻。"
)

# 防止从完全无关的帖子提取训练问答；短梗来源也需要电竞上下文。
GAME_CONTEXT_RE = re.compile(
    r"英雄联盟|\blol\b|\blpl\b|\blck\b|\bmsi\b|s赛|全球总决赛|世界赛|季中赛|电竞|"
    r"无畏契约|瓦洛兰特|\bvalorant\b|\bvct\b|冠军赛|大师赛|"
    r"排位|上单|中单|打野|辅助|\badc\b|ad位|对线|补刀|团战|线权|"
    r"盲僧|亚索|阿卡丽|厄斐琉斯|薇恩|\bvn\b|剑魔|塞拉斯|德莱文|"
    r"打大龙|小龙|峡谷|\bbp\b|ban位|\bbo3\b|\bbo5\b|"
    r"技能|eco局|强起|半起|爆头|残局|补枪|下包|拆包|"
    r"faker|uzi|theshy|\bbin\b|knight|chovy|rookie|xiaohu|"
    r"\bedg\b|\bblg\b|\btes\b|\bjdg\b|\big\b|\brng\b|\bfpx\b|\bwbg\b|\bt1\b|geng|gen\.g|\bhle\b|\bdk\b|"
    r"\bprx\b|fnatic|sentinels|\bnrg\b|康康|zmjjkk|smoggy|whzy", re.I
)
BLOCK_RE = re.compile(
    r"https?://|www\.|(?:t\.cn|bit\.ly)/|"
    r"京东|拼多多|淘宝|购物礼金|优惠礼包|优惠券|返现|领红包|"
    r"扫码|加微|加v|私聊购买|代练接单|购买链接|关注公众号|"
    r"QQ群[:：]?\s*\d{5,}|微信[:：]?|vx[:：]?|v信|"
    r"\b1[3-9]\d{9}\b|[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}", re.I
)
RISK_RE = re.compile(
    r"开盒|人肉搜索|身份证号|家庭住址|电话号码|杀你全家|死全家|自杀去吧|去死吧|种族灭绝"
)
LOW_CONTENT_RE = re.compile(
    r"^(?:[?？!！。\s]+|[哈啊呃哦嗯啧哎]{3,}|666+|\d+|顶|插眼|蹲|前排|cy|dd|mark|支持|确实)$",
    re.I,
)
# 对有语境的短回复不再按长度一刀切；匹配只代表“值得审核”，不是高质量保证。
SHORT_MEME_WORDS = {
    "典", "绷", "乐", "急", "急了", "破防", "笑死", "难绷", "闹麻了",
    "伟大", "幽默", "逆天", "纯路人", "好好好", "别急", "贷款", "清算",
    "红温", "下饭", "坐牢", "世一", "神中神", "就这", "有一说一",
    "没绷住", "太典了", "典中典", "赢麻了", "输麻了", "又开始了",
    "纯在串", "这也能吹", "鉴定为", "你说得对", "怎么说话", "又贷上了",
    "不如", "尬黑", "串完了", "别尬黑", "真别尬黑",
}
SHORT_ENDING_RE = re.compile(r"[!！?？。.～~\s]+$")
STYLE_WORDS = (
    "是吧", "闹麻了", "难绷", "贷款", "清算", "逆天", "纯路人",
    "世一", "幽默", "嘴硬", "回旋镖", "绷不住", "好好好",
    "坐牢", "红温", "伟大", "真有你的", "这下", "又开始", "典中典",
    "鉴定为", "尬黑", "赢麻", "输麻", "开香槟",
)

REVIEW_FIELDS = [
    "approve", "id", "post_id", "score", "corpus_freq", "post_freq",
    "title", "body_excerpt", "prompt", "completion", "source_url",
]
REJECT_FIELDS = [
    "reason", "reason_zh", "post_id", "title", "body_excerpt", "reply",
    "source_url", "length",
]
SHORT_FIELDS = [
    "approve", "meme", "corpus_freq", "post_freq", "category",
    "example_prompt", "example_title", "example_post_id",
    "example_source_url", "other_contexts",
]
REASONS = {
    "not_gaming": "非游戏相关帖子（抽样）",
    "bad_context": "帖子标题/主楼含风险或标题长度异常（已隐藏）",
    "bad_length": "回复过短或超过75字",
    "spam_or_risky": "广告、联系信息或高风险内容（已隐藏）",
    "low_content": "低信息量、灌水或纯数字",
    "symbols": "符号比例过高",
    "duplicates": "重复内容（频率已另外统计）",
    "per_post_limit": "同帖候选超出每帖配额",
}


def normalized(text: str) -> str:
    return unicodedata.normalize("NFKC", text).strip()


def clean_text(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return re.sub(r"\s+", " ", normalized(value)).strip()


def duplicate_key(text: str) -> str:
    return re.sub(r"[\W_]+", "", normalized(text).casefold())


def is_blocked(text: str) -> bool:
    return bool(BLOCK_RE.search(text) or RISK_RE.search(text))


def spreadsheet_safe(text: object) -> object:
    """防止WPS/Excel打开公开论坛内容时执行CSV公式。"""
    if isinstance(text, str) and text.lstrip().startswith(("=", "+", "-", "@")):
        return "'" + text
    return text


def spreadsheet_restore(text: str) -> str:
    if text.startswith("'") and len(text) > 1 and text[1] in "=+-@":
        return text[1:]
    return text


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: spreadsheet_safe(row.get(k, "")) for k in fieldnames})


def backup_and_approvals(path: Path, id_col: str) -> dict[str, str]:
    """重新extract不会覆盖用户已有的审核结果；原CSV先备份。"""
    if not path.exists():
        return {}
    approvals = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            key = spreadsheet_restore(str(row.get(id_col, "")))
            if key:
                approvals[key] = str(row.get("approve", "")).strip()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    dest = path.with_name(f"{path.stem}_backup_{timestamp}{path.suffix}")
    shutil.copy2(path, dest)
    print("已自动备份旧审核表:", dest)
    return approvals


def is_short_meme(text: str) -> bool:
    if is_blocked(text) or len(text) > 9:
        return False
    tidy = SHORT_ENDING_RE.sub("", text).strip()
    return tidy in SHORT_MEME_WORDS


def reply_quality(reply: str, title: str) -> int:
    score = 0
    length = len(reply)
    if 6 <= length <= 28:
        score += 3
    elif length <= 45:
        score += 2
    elif length <= 58:
        score += 1
    score += sum(1 for word in STYLE_WORDS if word in reply)
    if any(ch in reply for ch in "？?！!"):
        score += 1
    title_chars = {title[i:i + 2] for i in range(max(0, len(title) - 1))}
    reply_chars = {reply[i:i + 2] for i in range(max(0, len(reply) - 1))}
    if len(title_chars & reply_chars) >= 2:
        score += 1
    return score


def create_prompt(title: str, body: str) -> str:
    if body and 8 <= len(body) <= 100 and duplicate_key(body) != duplicate_key(title):
        return f"{title}\n{body}"
    return title


def cmd_extract(args: argparse.Namespace) -> None:
    paths = sorted(SOURCE_DIR.rglob("tieba_post_detail_page_*.json"))
    if not paths:
        raise SystemExit(f"未找到原始JSON：{SOURCE_DIR}")
    if args.max_per_post < 1 or args.rejected_per_reason < 0 or args.min_short_count < 1:
        raise SystemExit("参数错误：max-per-post>=1, rejected-per-reason>=0, min-short-count>=1")

    # 先读审核状态，再扫描。只有全部扫描成功才会覆盖旧表。
    old_review = backup_and_approvals(REVIEW_FILE, "id")
    old_short = backup_and_approvals(SHORT_FILE, "meme")
    stats: Counter[str] = Counter()
    rng = random.Random(args.seed)
    rejected: dict[str, list[dict]] = defaultdict(list)
    reject_seen: Counter[str] = Counter()
    seen_normal: set[str] = set()
    all_freq: Counter[str] = Counter()
    normal_posts: dict[str, set[str]] = defaultdict(set)
    short_data: dict[str, dict] = {}
    candidates: list[dict] = []

    def reject(reason: str, post_id: str, title: str, body: str,
               reply: str, url: str) -> None:
        stats[reason] += 1
        reject_seen[reason] += 1
        masked = reason in ("spam_or_risky", "bad_context")
        sample = {
            "reason": reason, "reason_zh": REASONS[reason], "post_id": post_id,
            "title": "[内容已隐藏]" if masked else title[:110],
            "body_excerpt": "[内容已隐藏]" if masked else body[:100],
            "reply": "[广告/隐私/高风险内容已隐藏]" if masked else reply[:180],
            "source_url": "" if masked else url,
            "length": len(reply),
        }
        # 逐类蓄水池抽样；0=导出全部被拒项。重复太多时CSV仍可控。
        buffer = rejected[reason]
        limit = args.rejected_per_reason
        if limit == 0 or len(buffer) < limit:
            buffer.append(sample)
        else:
            j = rng.randrange(reject_seen[reason])
            if j < limit:
                buffer[j] = sample

    for path in paths:
        try:
            posts = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            stats["read_error"] += 1
            print(f"[WARN] 跳过无法解析的文件：{path} ({exc})")
            continue
        if not isinstance(posts, list):
            stats["invalid_json_shape"] += 1
            continue
        stats["files"] += 1
        for index, post in enumerate(posts):
            stats["posts"] += 1
            if not isinstance(post, dict):
                stats["invalid_post"] += 1
                continue
            title = clean_text(post.get("标题"))
            body = clean_text(post.get("楼主内容"))
            url = clean_text(post.get("链接"))
            replies = post.get("回复列表")
            post_id = hashlib.sha1(f"{path.name}:{index}:{url}".encode()).hexdigest()[:16]
            if not 5 <= len(title) <= 110 or is_blocked(title) or is_blocked(body[:250]):
                # 旧版对此类帖子的回复不逐条统计；这里只保留一条脱敏样本。
                reject("bad_context", post_id, title, body, "", url)
                continue
            if not GAME_CONTEXT_RE.search(title + " " + body[:260]):
                # 非游戏帖子不逐条储存个人发言；仅抽取首条说明被排除的上下文。
                first = clean_text(replies[0]) if isinstance(replies, list) and replies else ""
                if is_blocked(first):
                    first = "[内容已隐藏]"
                reject("not_gaming", post_id, title, body, first, url)
                continue
            if not isinstance(replies, list):
                stats["invalid_replies"] += 1
                continue

            prompt = create_prompt(title, body)
            within_post: list[tuple[int, str, str]] = []
            for reply in replies:
                stats["replies_seen"] += 1
                text = clean_text(reply)
                if is_blocked(text):
                    reject("spam_or_risky", post_id, title, body, text, url)
                    continue
                if len(re.findall(r"[^\w\u4e00-\u9fff\s]", text)) > max(12, len(text) // 2):
                    reject("symbols", post_id, title, body, text, url)
                    continue
                if LOW_CONTENT_RE.fullmatch(text) or not duplicate_key(text):
                    reject("low_content", post_id, title, body, text, url)
                    continue

                key = duplicate_key(text)
                # 符合语境的短梗，无论重复多少次都累加频次。
                if is_short_meme(text):
                    stats["short_meme_occurrences"] += 1
                    info = short_data.setdefault(key, {
                        "meme": text, "count": 0, "post_ids": set(), "contexts": [],
                    })
                    info["count"] += 1
                    if post_id not in info["post_ids"]:
                        info["post_ids"].add(post_id)
                        if len(info["contexts"]) < 3:
                            info["contexts"].append({
                                "prompt": prompt, "title": title, "post_id": post_id,
                                "source_url": url,
                            })
                    # <=5字：旧脚本本来会丢掉，这里专门进入短梗表。
                    if len(text) < 6:
                        continue
                    # 6-9字可同时进入普通审核表，但build有查重保护。

                if not 6 <= len(text) <= 75:
                    reject("bad_length", post_id, title, body, text, url)
                    continue
                if len(key) < 5:
                    reject("low_content", post_id, title, body, text, url)
                    continue
                all_freq[key] += 1
                normal_posts[key].add(post_id)
                if key in seen_normal:
                    reject("duplicates", post_id, title, body, text, url)
                    continue
                seen_normal.add(key)
                within_post.append((reply_quality(text, title), text, key))
            within_post.sort(key=lambda x: (-x[0], len(x[1])))
            for score, text, key in within_post[:args.max_per_post]:
                row_id = hashlib.sha1((post_id + "\0" + text).encode()).hexdigest()[:20]
                candidates.append({
                    "approve": old_review.get(row_id, ""), "id": row_id,
                    "post_id": post_id, "score": score,
                    "corpus_freq": 0, "post_freq": 0,
                    "title": title, "body_excerpt": body[:120],
                    "prompt": prompt, "completion": text, "source_url": url,
                    "_key": key,
                })
            for _, text, _ in within_post[args.max_per_post:]:
                reject("per_post_limit", post_id, title, body, text, url)

    for row in candidates:
        key = row.pop("_key")
        row["corpus_freq"] = all_freq[key]
        row["post_freq"] = len(normal_posts[key])
        # 同一评论反复刷屏不代表全网热度；优先略参考跨帖子出现。
        row["score"] += min(2, max(0, row["post_freq"] - 1))
    candidates.sort(key=lambda r: (-int(r["score"]), -int(r["post_freq"]), r["post_id"], r["id"]))

    short_rows: list[dict] = []
    for info in short_data.values():
        if info["count"] < args.min_short_count:
            continue
        contexts = info["contexts"]
        if not contexts:
            continue
        best = contexts[0]
        short_rows.append({
            "approve": old_short.get(info["meme"], ""),
            "meme": info["meme"], "corpus_freq": info["count"],
            "post_freq": len(info["post_ids"]), "category": "短梗候选-需审核",
            "example_prompt": best["prompt"], "example_title": best["title"],
            "example_post_id": best["post_id"], "example_source_url": best["source_url"],
            "other_contexts": " | ".join(x["title"] for x in contexts[1:]),
        })
    short_rows.sort(key=lambda r: (-int(r["post_freq"]), -int(r["corpus_freq"]), r["meme"]))
    rejected_rows = []
    for reason in sorted(rejected):
        rejected_rows.extend(rejected[reason])

    write_csv(REVIEW_FILE, REVIEW_FIELDS, candidates)
    write_csv(REJECT_FILE, REJECT_FIELDS, rejected_rows)
    write_csv(SHORT_FILE, SHORT_FIELDS, short_rows)

    summary = {
        "counts": dict(stats), "normal_candidates": len(candidates),
        "short_meme_types": len(short_rows), "rejected_exported": len(rejected_rows),
        "rejected_export_mode": "all" if args.rejected_per_reason == 0 else f"reservoir_max_{args.rejected_per_reason}_per_reason",
        "notes": [
            "short-meme popularity counts refer only to this sampled corpus, not whole-site trend rankings",
            "source replies may respond to other floors; samples require manual review",
            "blocked/risky contents are masked in rejected file",
        ],
    }
    STATS_FILE.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"扫描JSON: {stats['files']} 个; 帖子: {stats['posts']}; 游戏帖回复扫描: {stats['replies_seen']}")
    print(f"普通候选: {len(candidates)} -> {REVIEW_FILE}")
    print(f"短梗类型: {len(short_rows)}; 出现次数: {stats['short_meme_occurrences']} -> {SHORT_FILE}")
    print(f"拒绝样本导出: {len(rejected_rows)} -> {REJECT_FILE}")
    print(f"详细统计: {STATS_FILE}")
    print("筛选统计:", dict(stats))
    print("请审核 CSV 的 approve 列填写 y；短梗默认不会自动进入训练集。")


def approved_csv_rows(path: Path, prompt_col: str, completion_col: str,
                      post_col: str) -> tuple[int, list[dict]]:
    if not path.exists():
        return 0, []
    rows = []
    count = 0
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for raw in csv.DictReader(f):
            count += 1
            if str(raw.get("approve", "")).strip().lower() not in {"y", "yes", "1", "是", "通过"}:
                continue
            prompt = clean_text(spreadsheet_restore(raw.get(prompt_col, "") or ""))
            completion = clean_text(spreadsheet_restore(raw.get(completion_col, "") or ""))
            if not prompt or not completion or is_blocked(prompt) or is_blocked(completion):
                continue
            post_id = str(raw.get(post_col, "")).strip()
            if not post_id:
                continue
            rows.append({"post_id": post_id, "prompt": prompt, "completion": completion})
    return count, rows


def cmd_build(args: argparse.Namespace) -> None:
    if not REVIEW_FILE.exists():
        raise SystemExit(f"先运行 extract，并审核：{REVIEW_FILE}")
    total, approved = approved_csv_rows(REVIEW_FILE, "prompt", "completion", "post_id")
    normal_count = len(approved)
    short_accepted = 0
    if args.include_short:
        short_total, short_rows = approved_csv_rows(
            SHORT_FILE, "example_prompt", "meme", "example_post_id"
        )
        # 限制短梗所占比例，防止“典/绷”把正常语境回复挤掉。
        limit = math.floor(normal_count * args.short_max_fraction / max(0.01, 1 - args.short_max_fraction))
        if normal_count == 0:
            limit = 0
        short_rows = short_rows[:limit]
        short_accepted = len(short_rows)
        approved.extend(short_rows)
        print(f"短梗审核表 {short_total} 行, 本次加入 {short_accepted} 条 (上限占比 {args.short_max_fraction:.0%})")

    # 防止短梗和普通候选重复训练；同一个帖子的样本必须一起划分。
    unique, seen_pairs = [], set()
    for row in approved:
        key = (duplicate_key(row["prompt"]), duplicate_key(row["completion"]))
        if key not in seen_pairs:
            seen_pairs.add(key)
            unique.append(row)
    if len(unique) < 2:
        raise SystemExit(f"审核通过的有效样本只有 {len(unique)} 条，至少需要2条。")
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in unique:
        groups[row["post_id"]].append(row)
    if len(groups) < 2:
        raise SystemExit("至少需要来自两个不同帖子的已审核样本，以避免跨训练/验证集泄漏。")
    rng = random.Random(args.seed)
    post_ids = sorted(groups)
    rng.shuffle(post_ids)
    n_valid = min(len(post_ids) - 1, max(1, round(len(post_ids) * args.valid_fraction)))
    valid_ids = set(post_ids[:n_valid])
    train, valid = [], []
    for post_id, rows in groups.items():
        dest = valid if post_id in valid_ids else train
        for row in rows:
            dest.append({
                "prompt": [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": row["prompt"]},
                ],
                "completion": [{"role": "assistant", "content": row["completion"]}],
            })
    rng.shuffle(train)
    rng.shuffle(valid)
    for path, items in ((TRAIN_FILE, train), (VALID_FILE, valid)):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for item in items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"普通候选 {total} | 普通审核批准 {normal_count} | 短梗加入 {short_accepted}")
    print(f"去重后 {len(unique)} 条, 不同帖子 {len(groups)}")
    print(f"训练 {len(train)} 条 -> {TRAIN_FILE}")
    print(f"验证 {len(valid)} 条 -> {VALID_FILE}")
    print("训练/验证按帖子划分。两类审核都需核对上下文及授权。")


def main() -> None:
    parser = argparse.ArgumentParser(description="抗吧语料清洗V2：短梗/拒绝样本/人工审核/SFT")
    sub = parser.add_subparsers(dest="mode", required=True)
    extract = sub.add_parser("extract", help="更新候选审核表、短梗表及拒绝样本表")
    extract.add_argument("--max-per-post", type=int, default=3, help="每帖最多普通候选，默认3")
    extract.add_argument("--rejected-per-reason", type=int, default=2000,
                         help="每类拒绝原因最多导出多少条随机样本，0表示全部，默认2000")
    extract.add_argument("--min-short-count", type=int, default=1,
                         help="短梗至少在语料里出现几次才进入短梗表，默认1")
    extract.add_argument("--seed", type=int, default=3407)
    extract.set_defaults(func=cmd_extract)
    build = sub.add_parser("build", help="从approve=y审核行生成训练/验证数据")
    build.add_argument("--seed", type=int, default=3407)
    build.add_argument("--valid-fraction", type=float, default=0.1)
    build.add_argument("--include-short", action="store_true",
                       help="额外把短梗表中approve=y的样本（有比例上限）纳入训练")
    build.add_argument("--short-max-fraction", type=float, default=0.15,
                       help="短梗最多占训练样本总数比例，默认0.15")
    build.set_defaults(func=cmd_build)
    args = parser.parse_args()
    if args.mode == "build" and not (0 < args.valid_fraction < 1 and 0 <= args.short_max_fraction < 1):
        parser.error("valid-fraction须介于0和1，short-max-fraction须介于0和1（含0）")
    args.func(args)


if __name__ == "__main__":
    main()
