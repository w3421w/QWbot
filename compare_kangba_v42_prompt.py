#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""QWbot V4.2: 同一个V4 LoRA的旧/新系统提示词匿名A/B评测。

运行（不训练，不修改LoRA）：
 python compare_kangba_v42_prompt.py --check-only
 python compare_kangba_v42_prompt.py --limit 12 --repeats 2

给盲测CSV打分后：
 python compare_kangba_v42_prompt.py --summarize

使用已存在的 datasets/kangba_v41_new_holdout.csv；保留测试题作为留出集，禁止加入训练。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", str(ROOT / "hf_cache"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

MODEL_NAME = "unsloth/Qwen3-1.7B-unsloth-bnb-4bit"
ADAPTER = ROOT / "outputs" / "kangba_v4_lora"
QUESTIONS = ROOT / "datasets" / "kangba_v41_new_holdout.csv"
OUTPUT = ROOT / "outputs"
PREFIX = "kangba_v42_prompt"

# 旧版与上一轮V4.1评测的SYSTEM完全相同，保证可以对照。
OLD_SYSTEM = (
    "你是QQ群里的普通群友，说话带抗压背锅吧那种简短、抽象、反串、回旋镖的味道。"
    "先理解对方在说什么，接得上话才整活；不用客服腔，不长篇说教，不硬塞游戏梗。"
    "抗吧常聊电竞但不限电竞，日常水帖也能正常聊；真实事件不编造，严肃求助时正常回应。"
)

# 修订后的V4.2提示词：允许自然使用单字短梗，但不强制使用。
NEW_SYSTEM = (
    "你是QQ群里长期混抗压背锅吧的普通群友，说话自然、抽象，擅长反串、阴阳怪气、回旋镖和接梗，"
    "但不是客服，也不是复读机。\n\n"
    "先理解群友的具体意思和情绪，再决定怎么回应。遇到贷款吹，可以顺势反串或拆台；"
    "遇到双标、嘴硬和前后矛盾，可以用回旋镖；遇到日常水帖，就自然接话。\n\n"
    "允许根据情绪和语境自然使用抗吧常见表达，例如‘典’‘绷’‘闹麻了’‘急了’‘孝’‘差不多得了’等。"
    "情绪强烈或笑点明确时，可以只回复一个字或几个字；也可以回复完整的一句话。"
    "不要为了显示抗吧风格而机械堆砌黑话。\n\n"
    "回复通常简短，以自然、有梗、切题为先，不强制字数。不要写客服腔，不解释笑点，"
    "不无意义重复，也不要凭空编造聊天背景。\n\n"
    "涉及真实赛事、选手转会和其他需要查证的信息，不要冒充已经核实；"
    "有人认真求助、情绪低落或明确要求别开玩笑时，应正常回应。\n\n"
    "只输出要发给群友的回复，不输出分析或额外说明。"
)

COLUMNS = [
    "id", "category", "prompt", "reply_A", "reply_B",
    "which_is_better_A_B_tie",
    "relevance_A_0_2", "relevance_B_0_2",
    "style_A_0_2", "style_B_0_2",
    "coherence_A_0_2", "coherence_B_0_2",
    "fabrication_or_harm_A_y_n", "fabrication_or_harm_B_y_n",
    "notes",
]


def output_paths():
    return (
        OUTPUT / f"{PREFIX}_blind_scoring.csv",
        OUTPUT / f"{PREFIX}_blind_key.json",
        OUTPUT / f"{PREFIX}_raw_results.json",
    )


def read_questions(limit: int):
    if not QUESTIONS.exists():
        raise FileNotFoundError(f"缺少留出集：{QUESTIONS}\n请把V4.1工具包里的测试题CSV放入datasets目录")
    with QUESTIONS.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or not all((r.get("prompt") or "").strip() for r in rows):
        raise ValueError("留出集为空或有缺失prompt的行")
    categories = {}
    for row in rows:
        categories.setdefault(row.get("category") or "其他", []).append(row)
    balanced = []
    for j in range(max(map(len, categories.values()))):
        for arr in categories.values():
            if j < len(arr):
                balanced.append(arr[j])
    return balanced[:limit] if limit else balanced


def validate_args(args):
    if args.limit < 0 or not 1 <= args.repeats <= 5 or args.max_new_tokens < 1:
        raise ValueError("要求 limit>=0、repeats在1-5之间、max-new-tokens>0")
    if not (0 < args.temperature <= 2):
        raise ValueError("temperature应在(0,2]之间")


def summarize():
    scored, key_path, _ = output_paths()
    if not scored.exists() or not key_path.exists():
        raise FileNotFoundError("找不到匿名评分CSV或解盲映射JSON；请先完成测试")
    with scored.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    mapping = {m["id"]: m for m in json.loads(key_path.read_text(encoding="utf-8"))["mapping"]}
    names = ["V4旧提示词", "V4.2新提示词"]
    wins = Counter()
    per_category = defaultdict(Counter)
    scores = defaultdict(lambda: defaultdict(list))
    filled, invalid = 0, []
    for row in rows:
        pick = row["which_is_better_A_B_tie"].strip().lower()
        if not pick:
            continue
        if pick not in {"a", "b", "tie", "平局"}:
            invalid.append(f"{row['id']}: {pick}")
            continue
        if row["id"] not in mapping:
            invalid.append(f"{row['id']}: 解盲映射中不存在")
            continue
        m = mapping[row["id"]]
        winner = "平局" if pick in {"tie", "平局"} else m[pick.upper()]
        wins[winner] += 1
        per_category[row["category"]][winner] += 1
        filled += 1
        for side in ("A", "B"):
            name = m[side]
            for metric in ("relevance", "style", "coherence"):
                val = (row.get(f"{metric}_{side}_0_2") or "").strip()
                if val:
                    try:
                        n = float(val)
                        if 0 <= n <= 2:
                            scores[name][metric].append(n)
                        else:
                            invalid.append(f"{row['id']} {metric}_{side}: 超出0-2范围")
                    except ValueError:
                        invalid.append(f"{row['id']} {metric}_{side}: 非数字")
    print(f"已评分 {filled}/{len(rows)} 组", flush=True)
    for name in (*names, "平局"):
        print(f"{name}: {wins[name]} 组")
    total_non_tie = wins[names[0]] + wins[names[1]]
    if total_non_tie:
        print(f"V4.2非平局胜率: {wins[names[1]] / total_non_tie:.1%}（仅这批题，不代表真实群聊胜率）")
    print("\n每项平均分（0-2）：")
    for name in names:
        print(name, ", ".join(
            f"{metric}={sum(scores[name][metric]) / len(scores[name][metric]):.2f}"
            if scores[name][metric] else f"{metric}=未评分"
            for metric in ("relevance", "style", "coherence")
        ))
    print("\n分类胜负：")
    for category, counts in per_category.items():
        print(category, " / ".join(f"{name}:{counts[name]}" for name in (*names, "平局")))
    if invalid:
        print("\n以下项需要检查：", "; ".join(invalid[:15]))
    if filled != len(rows):
        print("提醒：评分未填完，以上为已有评分的阶段统计。")
    print("提示：涉及编造事实、认真求助等题目，要另外审查，不能仅靠总胜率判断可否部署。")


def run(args):
    validate_args(args)
    questions = read_questions(args.limit)
    counts = Counter(q.get("category") or "其他" for q in questions)
    print(f"准备评测：{len(questions)}题 × {args.repeats}轮 = {len(questions)*args.repeats}组")
    print("类别：", dict(counts))
    print("仅比较V4旧系统提示词与V4.2新系统提示词：同一模型、同一LoRA、同一组种子")
    if args.check_only:
        print("检查完成；未加载GPU模型，也未修改训练文件。")
        return

    if not (ADAPTER / "adapter_config.json").exists():
        raise FileNotFoundError(f"找不到V4 LoRA：{ADAPTER}")
    paths = output_paths()
    existing = [str(p) for p in paths if p.exists()]
    if existing:
        raise FileExistsError("为避免覆盖历史结果，请先备份或改名：\n" + "\n".join(existing))

    # Unsloth必须先于torch/peft/transformers导入。
    import unsloth  # noqa: F401
    import torch
    from unsloth import FastModel
    from peft import PeftModel

    if not torch.cuda.is_available():
        raise RuntimeError("没检测到CUDA；请在QWbot WSL GPU环境运行")
    print("加载原始Qwen3 1.7B 4bit + 已训练的V4 LoRA……", flush=True)
    model, tokenizer = FastModel.from_pretrained(
        model_name=MODEL_NAME, max_seq_length=512,
        dtype=torch.float16, load_in_4bit=True,
    )
    model = PeftModel.from_pretrained(model, str(ADAPTER), adapter_name="kangba_v4")
    model.eval()
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    eos = tokenizer.eos_token_id
    stop_ids = list(dict.fromkeys(i for i in (eos, im_end) if isinstance(i, int) and i >= 0))
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos
    if pad is None:
        pad = stop_ids[0]

    def generate(message: str, system: str, seed: int):
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": message},
        ]
        ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            enable_thinking=False, return_tensors="pt",
        ).to("cuda")
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        with torch.inference_mode():
            result = model.generate(
                input_ids=ids,
                max_length=ids.shape[-1] + args.max_new_tokens,
                do_sample=True,
                temperature=args.temperature,
                top_p=0.9,
                repetition_penalty=1.15,
                no_repeat_ngram_size=4,
                eos_token_id=stop_ids,
                pad_token_id=pad,
            )
        return tokenizer.decode(
            result[0, ids.shape[-1]:], skip_special_tokens=True
        ).strip() or "（空回复）"

    ratings, keys, raw = [], [], []
    for i, q in enumerate(questions, 1):
        question = q["prompt"]
        for r in range(1, args.repeats + 1):
            rid = f"T{i:03d}-R{r}"
            seed = 3407 + 101 * i + r
            old = generate(question, OLD_SYSTEM, seed)
            new = generate(question, NEW_SYSTEM, seed)
            swap = int(hashlib.sha256(rid.encode("utf-8")).hexdigest(), 16) % 2 == 0
            a, b = (new, old) if swap else (old, new)
            keys.append({
                "id": rid,
                "A": "V4.2新提示词" if swap else "V4旧提示词",
                "B": "V4旧提示词" if swap else "V4.2新提示词",
            })
            ratings.append({
                "id": rid, "category": q.get("category", ""),
                "prompt": question, "reply_A": a, "reply_B": b,
                **{field: "" for field in COLUMNS[5:]},
            })
            raw.append({
                "id": rid, "category": q.get("category", ""),
                "prompt": question, "old_system_reply": old,
                "v42_system_reply": new, "seed": seed,
            })
            print(f"{rid} [{q.get('category', '')}] 群友：{question}", flush=True)
            print(" A:", a[:240], flush=True)
            print(" B:", b[:240], flush=True)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    # 已检查过冲突；结果文件只在全部生成成功后写入。
    with paths[0].open("x", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(ratings)
    paths[1].write_text(json.dumps({
        "purpose": "必须先完成盲评再看此文件",
        "mapping": keys,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    paths[2].write_text(json.dumps({
        "old_system": OLD_SYSTEM, "new_system": NEW_SYSTEM,
        "generation": {
            "repeats": args.repeats, "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature, "top_p": 0.9,
            "repetition_penalty": 1.15, "no_repeat_ngram_size": 4,
        },
        "records": raw,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n完成！匿名评分CSV:", paths[0], "\n解盲文件:", paths[1],
          "\n原始生成结果:", paths[2])
    print("先盲评，评分完成后执行：python compare_kangba_v42_prompt.py --summarize")


def main():
    p = argparse.ArgumentParser(description="QWbot V4.2提示词匿名A/B：同一个V4 LoRA，不重新训练")
    p.add_argument("--limit", type=int, default=12, help="按场景均衡选题数量；0表示全部")
    p.add_argument("--repeats", type=int, default=2, help="每题生成次数（1-5）")
    p.add_argument("--max-new-tokens", type=int, default=48)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--check-only", action="store_true", help="仅读取测试题并展示分布")
    p.add_argument("--summarize", action="store_true", help="根据已评分CSV与解盲文件统计结果（不加载GPU）")
    args = p.parse_args()
    if args.summarize:
        summarize()
    else:
        run(args)


if __name__ == "__main__":
    main()
