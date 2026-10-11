#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""QWbot: V4.3 vs V4.4 adapter-only blind comparison (never retrains models).

python compare_kangba_v44.py --check-only
python compare_kangba_v44.py --limit 16 --repeats 2
# Fill in outputs/kangba_v44_blind_scoring.csv before looking at the key!
python compare_kangba_v44.py --summarize
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", str(ROOT / "hf_cache"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

MODEL_NAME = "unsloth/Qwen3-1.7B-unsloth-bnb-4bit"
V43_PATH = ROOT / "outputs/kangba_v43_lora"
V44_PATH = ROOT / "outputs/kangba_v44_lora"
QUESTION_PATH = ROOT / "datasets/kangba_v44_holdout.csv"
TRAIN_PATH = ROOT / "datasets/kangba_v44_train.jsonl"
VALID_PATH = ROOT / "datasets/kangba_v44_valid.jsonl"
OUT_DIR = ROOT / "outputs"
PREFIX = "kangba_v44"

# Use one fixed prompt for BOTH LoRAs. The V4.3 and V4.4 train sets use the SAME system prompt.
FALLBACK_SYSTEM = (
    "你是QQ群里长期混抗压背锅吧的普通群友，不是客服。先听懂对方在说什么，再自然接话。"
    "可以用反串、回旋镖、阴阳怪气，也可以自然使用“典”“绷”“闹麻了”等短梗；该用才用，"
    "不要随机拼接黑话、硬套游戏话题或无意义复读。一般一句话就够，不强制字数；"
    "认真求助和明确不想被开玩笑时正常回应。新闻、赛事、转会等事实未知时，"
    "不编造、不假装已查过。只输出要发到QQ群的回复。"
)

CSV_FIELDS = [
    "id", "category", "prompt", "reply_A", "reply_B",
    "which_is_better_A_B_tie", "relevance_A_0_2", "relevance_B_0_2",
    "style_A_0_2", "style_B_0_2", "coherence_A_0_2", "coherence_B_0_2",
    "fabrication_or_harm_A_y_n", "fabrication_or_harm_B_y_n", "notes",
]


def paths():
    return (
        OUT_DIR / f"{PREFIX}_blind_scoring.csv",
        OUT_DIR / f"{PREFIX}_blind_key.json",
        OUT_DIR / f"{PREFIX}_raw_results.json",
    )


def normalize(s):
    return re.sub(r"\s+", "", str(s)).casefold()


def read_system():
    """Use exact V4.4 system text from the dataset if available."""
    for p in (TRAIN_PATH, VALID_PATH):
        if p.exists():
            with p.open("r", encoding="utf-8") as f:
                row = json.loads(next(line for line in f if line.strip()))
            text = row["prompt"][0]["content"]
            if text and row["prompt"][0]["role"] == "system":
                return text
    print("提示：没有找到V4.4训练文件，使用内置的V4.4系统提示词。")
    return FALLBACK_SYSTEM


def read_questions(limit):
    if not QUESTION_PATH.exists():
        raise FileNotFoundError(f"找不到测试题：{QUESTION_PATH}；请从V4.4工具包复制过来")
    with QUESTION_PATH.open("r", encoding="utf-8-sig", newline="") as f:
        data = list(csv.DictReader(f))
    if not data or any(not (x.get("prompt") or "").strip() for x in data):
        raise ValueError("留出集为空或缺少prompt")
    if len({normalize(x["prompt"]) for x in data}) != len(data):
        raise ValueError("留出集本身存在相同prompt")
    groups = defaultdict(list)
    for row in data:
        groups[row.get("category") or "其他"].append(row)
    # Round-robin: don't consume first category only.
    selected = []
    for index in range(max(map(len, groups.values()))):
        for group in groups.values():
            if index < len(group):
                selected.append(group[index])
    return selected[:limit] if limit else selected


def check_leakage(rows):
    prompts = [r["prompt"] for r in rows]
    training = []
    for path in (TRAIN_PATH, VALID_PATH):
        if not path.exists():
            print(f"注意：{path.name}不存在，无法进行训练集交叉检查")
            continue
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    obj = json.loads(line)
                    training.append(obj["prompt"][-1]["content"])
    exact = set(map(normalize, training)) & set(map(normalize, prompts))
    if exact:
        raise RuntimeError(f"测试题与训练/验证数据有{len(exact)}条完全重复，已取消测试")
    if training:
        suspicious = []
        for prompt in prompts:
            p = normalize(prompt)
            for old in training:
                o = normalize(old)
                if len(p) >= 12 and len(o) >= 12 and SequenceMatcher(None, p, o).ratio() >= 0.88:
                    suspicious.append((prompt, old))
                    break
        if suspicious:
            print(f"警告：发现{len(suspicious)}道与训练数据字面近似的测试题，建议复核后再把评测视为独立测试")
            for a, b in suspicious[:3]:
                print("  测试:", a[:75], "| 训练:", b[:75])
    return len(training)


def summarize():
    scored, key, _ = paths()
    if not scored.exists() or not key.exists():
        raise FileNotFoundError("找不到匿名评分CSV或解盲映射文件，先进行测试并评分")
    with scored.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    lookup = {r["id"]: r for r in json.loads(key.read_text(encoding="utf-8"))["mapping"]}
    winners = Counter()
    by_category = defaultdict(Counter)
    metrics = defaultdict(lambda: defaultdict(list))
    risk_count = Counter()
    scored_count = 0
    for row in rows:
        choose = (row.get("which_is_better_A_B_tie") or "").strip().lower()
        if not choose:
            continue
        if choose not in ("a", "b", "tie", "平局"):
            raise ValueError(f"{row['id']} 的胜负只能填 A/B/Tie")
        mapping = lookup[row["id"]]
        winner = "平局" if choose in ("tie", "平局") else mapping[choose.upper()]
        winners[winner] += 1
        by_category[row["category"]][winner] += 1
        scored_count += 1
        for side in ("A", "B"):
            name = mapping[side]
            for metric in ("relevance", "style", "coherence"):
                val = (row.get(f"{metric}_{side}_0_2") or "").strip()
                if val:
                    v = float(val)
                    if not 0 <= v <= 2:
                        raise ValueError(f"{row['id']} {metric}_{side}应为0-2")
                    metrics[name][metric].append(v)
            flagged = (row.get(f"fabrication_or_harm_{side}_y_n") or "").strip().lower()
            if flagged in ("y", "yes", "是"):
                risk_count[name] += 1
    print(f"\n已评分：{scored_count}/{len(rows)}组")
    for name in ("V4.3", "V4.4", "平局"):
        print(f"{name}：{winners[name]}组")
    decided = winners["V4.3"] + winners["V4.4"]
    if decided:
        print(f"V4.4有胜负样本胜率：{winners['V4.4']/decided:.1%}（不代表真实部署胜率）")
    for name in ("V4.3", "V4.4"):
        score_str = []
        for metric in ("relevance", "style", "coherence"):
            values = metrics[name][metric]
            score_str.append(f"{metric}={sum(values)/len(values):.3f}" if values else f"{metric}=未评分")
        print(name, ", ".join(score_str), f", 风险标记={risk_count[name]}")
    print("\n分类胜负：")
    for category, counts in by_category.items():
        print(category, "V4.3", counts["V4.3"], "V4.4", counts["V4.4"], "平局", counts["平局"])
    if scored_count < len(rows):
        print("还有未完成的评分；以上仅为已评分部分。")
    print("注意：事实查证和严肃求助需要分别过关，不能只看总胜率。")


def run(args):
    if args.limit < 0 or not 1 <= args.repeats <= 5 or args.max_new_tokens < 8:
        raise ValueError("limit必须>=0，repeats在1-5之间，max-new-tokens至少为8")
    if not 0 < args.temperature <= 2:
        raise ValueError("temperature应在(0,2]之间")
    questions = read_questions(args.limit)
    system = read_system()
    ntrain = check_leakage(questions)
    print(f"待测：{len(questions)}题 × {args.repeats}轮 = {len(questions)*args.repeats}组")
    print("场景分布：", dict(Counter(q.get("category", "其他") for q in questions)))
    print("训练+验证核对样本数：", ntrain)
    print("统一系统提示词：", system[:90] + ("..." if len(system) > 90 else ""))
    print("对比：V4.3 LoRA vs V4.4 LoRA；相同的Qwen3底座、提示词、采样参数及随机种子")
    if args.check_only:
        print("检查完成：未加载模型、未占用GPU、未写入测试结果")
        return
    for folder in (V43_PATH, V44_PATH):
        if not (folder / "adapter_config.json").exists():
            raise FileNotFoundError(f"缺少LoRA：{folder}")
    clashes = [str(p) for p in paths() if p.exists()]
    if clashes:
        raise FileExistsError("已有同名测试结果，为防覆盖请先备份/移动：\n" + "\n".join(clashes))

    import unsloth  # noqa: F401   Import BEFORE torch/peft/transformers.
    import torch
    from unsloth import FastModel
    from peft import PeftModel

    if not torch.cuda.is_available():
        raise RuntimeError("未检测到CUDA，请在有GPU的QWbot WSL环境运行")
    print("加载原始Qwen3 1.7B 4bit与两份LoRA……", flush=True)
    base, tokenizer = FastModel.from_pretrained(
        model_name=MODEL_NAME, max_seq_length=512,
        dtype=torch.float16, load_in_4bit=True,
    )
    model = PeftModel.from_pretrained(base, str(V43_PATH), adapter_name="v43", is_trainable=False)
    model.load_adapter(str(V44_PATH), adapter_name="v44", is_trainable=False)
    model.eval()
    end_tok = tokenizer.convert_tokens_to_ids("<|im_end|>")
    eos = tokenizer.eos_token_id
    stops = list(dict.fromkeys(x for x in (eos, end_tok) if isinstance(x, int) and x >= 0))
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos
    if pad is None:
        pad = stops[0]

    def generate(text, adapter, seed):
        model.set_adapter(adapter)
        ids = tokenizer.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": text}],
            tokenize=True, add_generation_prompt=True, enable_thinking=False,
            return_tensors="pt",
        ).to("cuda")
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        with torch.inference_mode():
            result = model.generate(
                input_ids=ids,
                max_length=ids.shape[-1] + args.max_new_tokens,
                do_sample=True, temperature=args.temperature, top_p=0.9,
                repetition_penalty=1.15, no_repeat_ngram_size=4,
                eos_token_id=stops, pad_token_id=pad,
            )
        return tokenizer.decode(result[0, ids.shape[-1]:], skip_special_tokens=True).strip() or "（空回复）"

    rows, mapping, raw = [], [], []
    for i, q in enumerate(questions, 1):
        for repeat in range(1, args.repeats + 1):
            rid = f"T{i:03d}-R{repeat}"
            seed = 3407 + i * 101 + repeat
            old = generate(q["prompt"], "v43", seed)
            new = generate(q["prompt"], "v44", seed)
            swap = int(hashlib.sha256(rid.encode()).hexdigest(), 16) % 2 == 0
            rep_a, rep_b = (new, old) if swap else (old, new)
            mapping.append({"id": rid, "A": "V4.4" if swap else "V4.3", "B": "V4.3" if swap else "V4.4"})
            rows.append({
                "id": rid, "category": q.get("category", ""), "prompt": q["prompt"],
                "reply_A": rep_a, "reply_B": rep_b,
                **{col: "" for col in CSV_FIELDS[5:]},
            })
            raw.append({"id": rid, "source_question_id": q.get("id"), "category": q.get("category", ""),
                        "prompt": q["prompt"], "v43_reply": old, "v44_reply": new, "seed": seed})
            print(f"\n{rid} [{q.get('category', '')}] 群友：{q['prompt']}", flush=True)
            print(" A:", rep_a[:250], flush=True)
            print(" B:", rep_b[:250], flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csvpath, keypath, rawpath = paths()
    with csvpath.open("x", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    keypath.write_text(json.dumps({"purpose": "完成盲评后再查看", "mapping": mapping}, ensure_ascii=False, indent=2), encoding="utf-8")
    rawpath.write_text(json.dumps({"system": system, "base_model": MODEL_NAME, "settings": {
        "max_new_tokens": args.max_new_tokens, "temperature": args.temperature,
        "top_p": 0.9, "repetition_penalty": 1.15, "no_repeat_ngram_size": 4,
        "repeats": args.repeats}, "records": raw}, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n完成！匿名评分CSV:", csvpath)
    print("解盲映射（请先不要查看）:", keypath)
    print("完整生成记录:", rawpath)
    print("评分完成后：python compare_kangba_v44.py --summarize")


def main():
    cli = argparse.ArgumentParser(description="抗吧QWbot V4.3 vs V4.4 同条件、同系统提示词匿名对比")
    cli.add_argument("--check-only", action="store_true", help="仅检查文件及测试题，不加载GPU")
    cli.add_argument("--summarize", action="store_true", help="盲评完成后，利用解盲表汇总，不加载GPU")
    cli.add_argument("--limit", type=int, default=16, help="按场景均衡选题数量；0表示全部")
    cli.add_argument("--repeats", type=int, default=2, help="每题回复次数")
    cli.add_argument("--max-new-tokens", type=int, default=48)
    cli.add_argument("--temperature", type=float, default=0.8)
    args = cli.parse_args()
    if args.summarize:
        summarize()
    else:
        run(args)


if __name__ == "__main__":
    main()
