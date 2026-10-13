#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从 V3 全量抗吧审核表中提取“短回复接梗”SFT候选，绝不改动原始文件。

命令：
  python select_kangba_style_v3.py extract
  # 在 datasets/kangba_style_candidates.csv 中人工将合格行 approve 填 y
  python select_kangba_style_v3.py build

特点：保留全部原始语料；此处仅为独立训练子集；保留审计信息和来源；按帖子划分验证集；
      限制相同回复在训练数据中的出现次数；把验证集里的答案和训练集撞车的项目剔除。
注意：自动打分不能理解楼中楼上下文；只会生成“候选”，绝不默认审核通过。
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import random
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "datasets"
SOURCE = DATA / "kangba_review.csv"
CANDIDATES = DATA / "kangba_style_candidates.csv"
TRAIN = DATA / "kangba_style_train.jsonl"
VALID = DATA / "kangba_style_valid.jsonl"
STATS = DATA / "kangba_style_stats.json"

SYSTEM = (
    "你是QQ群里的AI群友，懂LOL和无畏契约。"
    "日常先理解对方说什么，再用一句自然的短话接梗；可以反串和阴阳怪气。"
    "事实性问题不编造，别人明显难过时不反串。"
)

FIELDS = ["approve", "score", "reasons", "post_id", "id", "source_url", "prompt", "completion", "corpus_freq"]

GAME_TERMS = (
    "blg", "edg", "jdg", "tes", "ig", "rng", "nip", "gen", "t1", "lpl", "lck", "lol",
    "msi", "vct", "valorant", "faker", "elk", "bin", "uzi", "smoggy", "zmjjkk",
    "无畏契约", "瓦洛兰特", "打野", "上单", "中单", "下路", "辅助", "对线",
    "大龙", "小龙", "团战", "排位", "赛前", "赛后", "冠军", "世界赛", "打团",
    "补枪", "干拉", "残局", "选手", "队友", "连败", "直播", "红温", "贷款", "背锅",
)
STYLE_TERMS = (
    "是吧", "闹麻", "典", "绷", "笑死", "难绷", "幽默", "伟大", "逆天", "牢",
    "打脸", "清算", "贷款", "回旋镖", "世一", "纯路人", "下饭", "就这", "谁", "真", "还", "怎么",
)
# 这类问答容易带入未经核实的实时信息；从风格SFT暂时隔离，在机器人推理时交给知识库/Tavily。
CURRENT_EVENT = re.compile(r"(昨天|今天|刚刚|最新|是不是.*转会|官宣|出事了|哪天|几月|处罚|涉嫌|退役|签约)")
AD_SPAM = re.compile(r"(优惠礼包|购物礼金|京东首页搜|扫码.{0,6}领|返现|刷单|兼职日结|加.{0,2}微信|推广邀请码|点击链接|购买链接)", re.I)
PRIVATE = re.compile(r"(\[手机号已脱敏\]|\[邮箱已脱敏\]|\[联系方式已脱敏\]|\b1[3-9]\d{9}\b)")
THREAT = re.compile(r"(杀你全家|死全家|开盒.{0,6}(家人|地址)|种族灭绝)")
URL = re.compile(r"https?://|www\.", re.I)


def normalize(s: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", s)).strip()


def dedup_key(s: str) -> str:
    s = normalize(s).casefold()
    return re.sub(r"[^\w\u4e00-\u9fff]+", "", s)


def guarded_text(s: str) -> bool:
    return bool(AD_SPAM.search(s) or PRIVATE.search(s) or THREAT.search(s) or URL.search(s))


def feature_score(prompt: str, completion: str) -> tuple[int, list[str]]:
    """粗排而不是自动判定语义相关：楼层关系丢失时必须人工复核。"""
    p, c = normalize(prompt), normalize(completion)
    if not p or not c:
        return -100, ["empty"]
    if guarded_text(p) or guarded_text(c):
        return -100, ["spam_private_or_threat"]
    if CURRENT_EVENT.search(p) and ("?" in p or "？" in p or "吗" in p or "转会" in p):
        return -50, ["time_sensitive_fact_question"]
    if len(p) < 6:
        return -20, ["prompt_too_short"]
    if len(c) > 55:
        return -20, ["answer_over_55_chars"]
    if len(c) < 2:
        return -20, ["single_char_for_meme_bank"]
    if len(p) > 220:
        return -20, ["context_over_220_chars"]
    if c == p or dedup_key(c) == dedup_key(p):
        return -20, ["echo_input"]
    score = 0
    reasons = []
    if 5 <= len(c) <= 30:
        score += 2
        reasons.append("natural_short_reply")
    elif len(c) <= 45:
        score += 1
        reasons.append("moderate_length")

    p_lower, c_lower = p.casefold(), c.casefold()
    p_games = {term for term in GAME_TERMS if term in p_lower}
    c_games = {term for term in GAME_TERMS if term in c_lower}
    common = p_games & c_games
    if common:
        score += min(4, 2 + len(common))
        reasons.append("shared_game_context:" + "/".join(sorted(common)[:3]))
    elif p_games and c_games:
        score += 1
        reasons.append("both_game_related_but_different_terms")
    elif not p_games and not c_games:
        reasons.append("not_recognized_as_game_topic")

    if any(x in c for x in STYLE_TERMS):
        score += 1
        reasons.append("casual_or_meme_phrase")

    # 仅少量重叠才有意义；中文字的通用虚词不能作为强相关证据。
    p_grams = {p[i:i+2] for i in range(max(0, len(p)-1)) if '\u4e00' <= p[i] <= '\u9fff' and '\u4e00' <= p[i+1] <= '\u9fff'}
    c_grams = {c[i:i+2] for i in range(max(0, len(c)-1)) if '\u4e00' <= c[i] <= '\u9fff' and '\u4e00' <= c[i+1] <= '\u9fff'}
    common_grams = {g for g in p_grams & c_grams if g not in {"这个", "那个", "就是", "不是", "什么", "怎么", "一样", "现在", "没有", "可以", "自己", "还有", "感觉", "真的"}}
    if len(common_grams) >= 2:
        score += 1
        reasons.append("shared_chinese_terms")
    # 跑题风险提示，只是人工提示，不靠这个做确证。
    if not common and not common_grams and not any(x in c for x in STYLE_TERMS):
        reasons.append("weak_context_overlap_review_carefully")
    return score, reasons


def read_review():
    if not SOURCE.is_file():
        raise SystemExit(f"缺少 {SOURCE}，请先执行 prepare_kangba_corpus.py extract")
    with SOURCE.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if (row.get("approve") or "").strip().lower() in {"n", "no", "0", "否"}:
                continue
            yield row


def read_old_approvals():
    result = {}
    if not CANDIDATES.is_file():
        return result
    with CANDIDATES.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            key = row.get("id") or hashlib.sha1((row.get("prompt", "")+"\n"+row.get("completion", "")).encode()).hexdigest()
            a = (row.get("approve") or "").strip().lower()
            if a in {"y", "n"}:
                result[key] = a
    return result


def extract(args):
    old_approvals = read_old_approvals()
    picked = []
    reasons = Counter()
    scanned = 0
    for row in read_review():
        scanned += 1
        p = normalize(row.get("prompt", ""))
        c = normalize(row.get("completion", ""))
        score, why = feature_score(p, c)
        for r in why:
            reasons[r.split(":", 1)[0]] += 1
        if score < args.min_score:
            continue
        rid = row.get("id") or hashlib.sha1((p + "\n" + c).encode()).hexdigest()
        picked.append({
            "approve": old_approvals.get(rid, ""),
            "score": score,
            "reasons": ";".join(why),
            "post_id": row.get("post_id", ""),
            "id": rid,
            "source_url": row.get("source_url", ""),
            "prompt": p,
            "completion": c,
            "corpus_freq": row.get("corpus_freq", ""),
        })
    # 把最高分的候选优先展示，源文件只读，永不改动。
    picked.sort(key=lambda row: (-row["score"], row["id"]))
    if args.limit > 0:
        picked = picked[:args.limit]
    DATA.mkdir(parents=True, exist_ok=True)
    with CANDIDATES.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(picked)
    STATS.write_text(json.dumps({"stage":"extract","input_approved_rows":scanned,"candidates":len(picked),"reason_counts":dict(reasons),"note":"候选不是已审核；只会在 build 时使用 approve=y 的行"}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"扫描保留语料：{scanned} 条")
    print(f"风格候选：{len(picked)} 条（自动评分不等于通过审核）")
    print(f"审核表：{CANDIDATES}")
    print("请在 approve 列给适合的样本填 y，不合适填 n，其他留空。")


def build(args):
    if not CANDIDATES.is_file():
        raise SystemExit("请先执行 extract")
    raw = []
    checked = 0
    with CANDIDATES.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if (row.get("approve") or "").strip().lower() != "y":
                continue
            checked += 1
            p, c = normalize(row.get("prompt", "")), normalize(row.get("completion", ""))
            score, _ = feature_score(p, c)
            if score < args.min_score:
                continue
            raw.append((row.get("post_id") or dedup_key(p), p, c, int(row.get("score") or 0)))
    if len(raw) < 20:
        raise SystemExit(f"目前只有 {len(raw)} 条有效人工批准样本，建议至少先审核20条用于管线测试")
    # 先高分排序后同一答案限权：防止高频模板形成复读偏好。
    raw.sort(key=lambda x: (-x[3], x[0], x[1]))
    reply_counts, prompt_counts, exact_pairs = Counter(), Counter(), set()
    selected = []
    for post_id, p, c, score in raw:
        rk, pk = dedup_key(c), dedup_key(p)
        if (pk, rk) in exact_pairs:
            continue
        if reply_counts[rk] >= args.max_per_reply or prompt_counts[pk] >= args.max_per_prompt:
            continue
        exact_pairs.add((pk, rk))
        reply_counts[rk] += 1
        prompt_counts[pk] += 1
        selected.append((post_id, p, c, score))
    # 按帖子ID确定组，确保同一主题不跨训练和验证。
    groups = sorted({row[0] for row in selected})
    if len(groups) < 2:
        raise SystemExit("有效帖子数少于2，无法独立划分训练集和验证集")
    rnd = random.Random(args.seed)
    rnd.shuffle(groups)
    valid_groups = set(groups[:max(1, round(len(groups)*args.valid_fraction))])
    train_rows, valid_rows = [], []
    for post_id, p, c, score in selected:
        record = {"prompt":[{"role":"system","content":SYSTEM},{"role":"user","content":p}],"completion":[{"role":"assistant","content":c}]}
        (valid_rows if post_id in valid_groups else train_rows).append((p,c,record))
    # 如果不同帖子里存在相同的 prompt 或 completion，将其从验证集中剔除，减少评估泄露。
    training_prompts = {dedup_key(p) for p,c,rec in train_rows}
    training_answers = {dedup_key(c) for p,c,rec in train_rows}
    valid_rows = [(p,c,rec) for p,c,rec in valid_rows if dedup_key(p) not in training_prompts and dedup_key(c) not in training_answers]
    for path, rows in [(TRAIN,train_rows),(VALID,valid_rows)]:
        with path.open("w", encoding="utf-8") as f:
            for _,_,item in rows:
                f.write(json.dumps(item, ensure_ascii=False)+"\n")
    stat = {"stage":"build","approved_rows":checked,"after_frequency_caps":len(selected),"train":len(train_rows),"valid":len(valid_rows),"max_per_reply":args.max_per_reply,"max_per_prompt":args.max_per_prompt,"note":"仅人工审核通过的风格样本；训练时仍应统一Qwen非思考模板，其他原始训练集未更改"}
    STATS.write_text(json.dumps(stat, ensure_ascii=False, indent=2), encoding="utf-8")
    print("人工通过:", checked, "| 去重限权后:", len(selected))
    print("风格训练集:",len(train_rows),TRAIN)
    print("风格验证集:",len(valid_rows),VALID)
    print("提示：数据集只训练风格；最新赛事、转会仍交给事实知识库/联网搜索。")


def main():
    ap = argparse.ArgumentParser(description="抗吧V3：独立风格子集，原始数据绝不改动")
    sub = ap.add_subparsers(dest="mode", required=True)
    e = sub.add_parser("extract", help="抽取待审核的短句/接梗候选")
    e.add_argument("--min-score",type=int,default=3)
    e.add_argument("--limit",type=int,default=6000,help="最多导出多少行供人工审核；0=不限制")
    e.set_defaults(fn=extract)
    b = sub.add_parser("build", help="用人工approve=y构建独立风格训练集")
    b.add_argument("--min-score",type=int,default=3)
    b.add_argument("--max-per-reply",type=int,default=2)
    b.add_argument("--max-per-prompt",type=int,default=5)
    b.add_argument("--valid-fraction",type=float,default=0.1)
    b.add_argument("--seed",type=int,default=3407)
    b.set_defaults(fn=build)
    args = ap.parse_args()
    args.fn(args)

if __name__ == "__main__":
    main()
