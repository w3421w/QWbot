#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""QWbot V4.4 context diagnosis: base raw vs LoRA raw vs LoRA + extractive context cards.

All generations are local inference. No training, QQ API, external factual retrieval, or answer key in prompts.

Examples:
  python compare_kangba_v44_context_triad.py --check-only
  python compare_kangba_v44_context_triad.py --limit 5 --repeats 1 --tag smoke
  python compare_kangba_v44_context_triad.py --limit 0 --repeats 2 --tag full
  python compare_kangba_v44_context_triad.py --limit 0 --repeats 2 --tag full --resume
  python compare_kangba_v44_context_triad.py --summarize --tag full
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", str(ROOT / "hf_cache"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
DATA_FILE = ROOT / "datasets/QWbot_V44_真实群聊影子测试_匿名31题.jsonl"
ADAPTER_DIR = ROOT / "outputs/kangba_v44_lora"
TRAIN_FILE = ROOT / "datasets/kangba_v44_train.jsonl"
BASE_MODEL = "unsloth/Qwen3-1.7B-unsloth-bnb-4bit"
OUTPUT_DIR = ROOT / "outputs"

SYSTEM_FALLBACK = (
    "你是QQ群里长期混抗压背锅吧的普通群友，不是客服。先听懂对方在说什么，再自然接话。"
    "可以用反串、回旋镖、阴阳怪气，也可以自然使用“典”“绷”“闹麻了”等短梗；该用才用，"
    "不要随机拼接黑话、硬套游戏话题或无意义复读。一般一句话就够，不强制字数；"
    "认真求助和明确不想被开玩笑时正常回应。新闻、赛事、转会等事实未知时，"
    "不编造、不假装已查过。只输出要发到QQ群的回复。"
)

MODES = ("BASE_RAW", "V44_RAW", "V44_CARDS")
RAW_FIELDS = [
    "id", "repeat", "mode", "category", "priority", "has_context", "seed", "incoming_message",
    "context", "extracted_cards", "model_reply", "time_seconds", "input_tokens", "output_tokens",
]
BLIND_FIELDS = [
    "id", "repeat", "category", "priority", "has_context", "context", "incoming_message",
    "reply_A", "reply_B", "reply_C", "evaluation_criteria",
    "winner_A_B_C_Tie", "understanding_A_0_2", "understanding_B_0_2", "understanding_C_0_2",
    "style_A_0_2", "style_B_0_2", "style_C_0_2",
    "naturalness_A_0_2", "naturalness_B_0_2", "naturalness_C_0_2",
    "risk_A_y_n", "risk_B_y_n", "risk_C_y_n", "notes",
]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_questions() -> list[dict]:
    if not DATA_FILE.is_file():
        raise FileNotFoundError(f"缺少31题：{DATA_FILE}；将ZIP解压到Qwen_Finetune根目录")
    rows = []
    for i, line in enumerate(DATA_FILE.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        obj = json.loads(line)
        needed = ("id", "category", "priority", "context", "incoming_message", "evaluation_criteria")
        if any(k not in obj for k in needed):
            raise ValueError(f"测试题第{i}行缺字段：{needed}")
        if not isinstance(obj["context"], list):
            raise ValueError(f"{obj['id']} context应是列表")
        if any(not isinstance(msg.get("speaker"), str) or not isinstance(msg.get("text"), str)
               for msg in obj["context"]):
            raise ValueError(f"{obj['id']} context必须含speaker和text")
        rows.append(obj)
    if len(rows) != 31 or len({r["id"] for r in rows}) != len(rows):
        raise ValueError(f"预期31条不重复测试题；实际{len(rows)}条，请核实ZIP文件")
    return rows


def get_system() -> str:
    if TRAIN_FILE.is_file():
        with TRAIN_FILE.open(encoding="utf-8-sig") as f:
            for line in f:
                if line.strip():
                    obj = json.loads(line)
                    msgs = obj.get("prompt", [])
                    if msgs and msgs[0].get("role") == "system" and msgs[0].get("content"):
                        return msgs[0]["content"]
                    break
    return SYSTEM_FALLBACK


def context_history(row: dict) -> str:
    return "\n".join(f"{x['speaker']}：{x['text']}" for x in row["context"])


def make_context_cards(row: dict) -> list[dict]:
    """Strictly extractive: source_quote is exact input context and hints derive from same quote.

    No reference replies, external information, or original QQ identity data are accessed.
    """
    cards: list[dict] = []
    for idx, message in enumerate(row["context"], 1):
        speaker = message["speaker"]
        quote = message["text"]
        tag = "群友发言"
        extracted = ""
        if "[图片" in quote:
            tag = "媒体占位符（无图像内容）"
        else:
            alias = re.search(r"(?:叫我|称呼我)[“\"'《]?([^”\"'》。，；\s]{1,15})[”\"'》]?", quote)
            topic = re.search(r"我说的是\s*([^。，！\s]{1,20})", quote)
            if alias:
                tag = "本人自述称呼"
                extracted = f"{speaker}提到可以称呼自己为「{alias.group(1)}」"
            elif quote.startswith(("不对", "不是", "我说的是")) or "猜错" in quote:
                tag = "更正或异议"
                if topic:
                    extracted = f"提及话题「{topic.group(1)}」"
            elif "说" in quote or "是" in quote:
                tag = "群内说法（未核实）"
            numbers = re.findall(r"(?<!\w)\d+(?:[-.]\d+)*(?!\w)", quote)
            if numbers:
                suffix = "原话数字：" + "、".join(dict.fromkeys(numbers))
                extracted = extracted + ("；" if extracted else "") + suffix
        cards.append({"order": idx, "speaker": speaker, "type": tag,
                      "source_quote": quote, "extract": extracted})
    # Detect explicit rebuttal only when the source itself contains a correction marker.
    return cards


def build_user_message(row: dict, use_cards: bool) -> str:
    history = context_history(row)
    if not history:
        return row["incoming_message"]
    if not use_cards:
        return (f"以下是群里的聊天记录（只是上下文）：\n{history}\n\n"
                f"当前群友发言：{row['incoming_message']}\n请只回复当前群友。")
    cards = make_context_cards(row)
    card_lines = []
    for c in cards:
        label = f"- {c['speaker']} [{c['type']}] 原话：「{c['source_quote']}」"
        if c["extract"]:
            label += f" | 提取：{c['extract']}"
        card_lines.append(label)
    return ("以下是从群聊历史提取的信息卡片；它们仅说明群友说过什么，不等于事实已核验：\n"
            + "\n".join(card_lines)
            + f"\n\n以下是同一段原始群聊记录（可核对）：\n{history}\n\n"
            + f"当前群友发言：{row['incoming_message']}\n请只回复当前群友。")


def validate_extractiveness(row: dict) -> None:
    """Every source quotation must be literally present in the context; no gold content allowed."""
    original = [x["text"] for x in row["context"]]
    for card in make_context_cards(row):
        if card["source_quote"] not in original:
            raise AssertionError(f"{row['id']} 卡片原话不在上下文")
        if card["extract"]:
            if "提到可以称呼自己为" in card["extract"]:
                alias = re.search(r"称呼自己为「([^」]+)」", card["extract"])
                if not alias or alias.group(1) not in card["source_quote"]:
                    raise AssertionError(f"{row['id']} 昵称提取失真")
            if "原话数字：" in card["extract"]:
                nums = card["extract"].split("原话数字：", 1)[-1].split("、")
                if any(n not in card["source_quote"] for n in nums):
                    raise AssertionError(f"{row['id']} 数字提取失真")


def paths(tag: str) -> dict[str, Path]:
    name = f"kangba_v44_context_triad_{tag}"
    return {
        "raw": OUTPUT_DIR / f"{name}_raw_long.csv",
        "blind": OUTPUT_DIR / f"{name}_blind_scoring.csv",
        "key": OUTPUT_DIR / f"{name}_blind_key.json",
        "meta": OUTPUT_DIR / f"{name}_metadata.json",
        "report": OUTPUT_DIR / f"{name}_summary.json",
    }


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def check_duplicate_training(questions: list[dict]) -> None:
    if not TRAIN_FILE.is_file():
        print("提示：没有V4.4训练集，无法进行原问句完全重复检查。")
        return
    prompts = set()
    for name in ("kangba_v44_train.jsonl", "kangba_v44_valid.jsonl"):
        path = ROOT / "datasets" / name
        if not path.is_file():
            continue
        with path.open(encoding="utf-8-sig") as f:
            for line in f:
                if line.strip():
                    obj = json.loads(line)
                    dialogs = obj.get("prompt", [])
                    if dialogs and isinstance(dialogs[-1], dict):
                        prompts.add(re.sub(r"\s+", "", dialogs[-1].get("content", "")))
    exact = [r["id"] for r in questions
             if len(r["incoming_message"].strip()) >= 6
             and re.sub(r"\s+", "", r["incoming_message"]) in prompts]
    if exact:
        print("注意：部分当前问句与旧训练/验证问题字面重复，请审慎解释结果：", exact)
    else:
        print("输入问题字面完全重复检查：未发现长度≥6字的重复（不代表语义绝对不重合）。")


def arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="三组同条件对照：原始Qwen/V4.4/V4.4+群聊提取卡片。仅离线推理。")
    p.add_argument("--check-only", action="store_true", help="检查输入和提取卡片；不加载GPU")
    p.add_argument("--summarize", action="store_true", help="评分填完后统计;不加载GPU")
    p.add_argument("--limit", type=int, default=0, help="测试前N题，0=全部31题")
    p.add_argument("--repeats", type=int, default=1, help="每题重复几次(1-5)")
    p.add_argument("--tag", default="full", help="输出文件标签(英文数字下划线横杠)")
    p.add_argument("--resume", action="store_true", help="因中断重启，只补齐未生成的样本")
    p.add_argument("--max-new-tokens", type=int, default=80)
    p.add_argument("--temperature", type=float, default=0.8)
    return p


def read_and_validate_scores(tag: str) -> dict:
    out = paths(tag)
    if not out["blind"].exists() or not out["key"].exists():
        raise FileNotFoundError("缺少blind_scoring.csv或blind_key.json，先运行推理。")
    rows = read_csv(out["blind"])
    mapping = {r["id"]: r for r in json.loads(out["key"].read_text(encoding="utf-8"))["mapping"]}
    metrics = defaultdict(lambda: defaultdict(list))
    risks = Counter()
    winner = Counter()
    by_category = defaultdict(Counter)
    by_context = defaultdict(Counter)
    metrics_by_context = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    filled = 0
    for r in rows:
        id_ = f"{r['id']}-R{r['repeat']}"
        m = mapping[id_]
        choice = (r["winner_A_B_C_Tie"] or "").strip().upper()
        if not choice:
            continue
        if choice not in ("A", "B", "C", "TIE"):
            raise ValueError(f"{id_} 胜负列应填A/B/C/Tie，实际是{choice}")
        filled += 1
        win = "TIE" if choice == "TIE" else m[choice]
        winner[win] += 1
        by_category[r["category"]][win] += 1
        group = "有上下文" if r.get("has_context") == "y" else "无上下文"
        by_context[group][win] += 1
        for letter in "ABC":
            model_name = m[letter]
            for kind in ("understanding", "style", "naturalness"):
                value = (r.get(f"{kind}_{letter}_0_2") or "").strip()
                if value:
                    score = float(value)
                    if not 0 <= score <= 2:
                        raise ValueError(f"{id_} {kind}_{letter}分数必须在0-2")
                    metrics[model_name][kind].append(score)
                    metrics_by_context[group][model_name][kind].append(score)
            risk = (r.get(f"risk_{letter}_y_n") or "").strip().lower()
            if risk in ("y", "yes", "是"):
                risks[model_name] += 1
            elif risk and risk not in ("n", "no", "否"):
                raise ValueError(f"{id_} 风险字段应为y/n")
    return {
        "total_questions_repeats": len(rows), "rated": filled,
        "winners": dict(winner),
        "by_category": {k: dict(v) for k, v in by_category.items()},
        "by_has_context": {k: dict(v) for k, v in by_context.items()},
        "means_by_has_context": {g: {m: {t: round(sum(v)/len(v), 4) for t, v in ts.items() if v}
                                     for m, ts in models.items()}
                                 for g, models in metrics_by_context.items()},
        "averages_0_2": {k: {t: round(sum(v)/len(v), 4) for t, v in types.items() if v}
                         for k, types in metrics.items()},
        "risk_flags": dict(risks),
        "caution": "人工主观评分；三组生成对比不等于真实部署表现。卡片组同时添加了结构化提示，不能单独归因于持久记忆。",
    }


def summarize(tag: str):
    report = read_and_validate_scores(tag)
    file = paths(tag)["report"]
    if file.exists():
        raise FileExistsError(f"统计文件已存在：{file}，请先备份或改名")
    file.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("统计写入", file)


def main():
    args = arg_parser().parse_args()
    if args.limit < 0 or args.repeats not in range(1, 6):
        raise ValueError("--limit>=0, --repeats 1..5")
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,32}", args.tag):
        raise ValueError("--tag 必须是英文数字/短横线/下划线")
    if not 0 < args.temperature <= 2 or not 8 <= args.max_new_tokens <= 256:
        raise ValueError("--temperature在(0,2], --max-new-tokens在8-256")
    if args.summarize:
        return summarize(args.tag)

    all_questions = read_questions()
    questions = all_questions[:args.limit] if args.limit else all_questions
    for q in questions:
        validate_extractiveness(q)
        if "example_reply_for_human_review" in build_user_message(q, True):
            raise AssertionError("生成提示词包含参考答案字段")

    system = get_system()
    fp = sha256_bytes(json.dumps({
        "questions": questions, "system": system, "base": BASE_MODEL,
        "modes": MODES, "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature, "top_p": 0.9, "repetition_penalty": 1.15,
        "no_repeat_ngram_size": 4, "repeats": args.repeats,
    }, ensure_ascii=False, sort_keys=True).encode("utf-8"))
    total = len(questions) * args.repeats * len(MODES)
    print(f"题目数: {len(questions)}；每题{args.repeats}次；模式数:3；总生成:{total}")
    print(f"底座: {BASE_MODEL}\n适配器: {ADAPTER_DIR}\n系统提示词hash: {sha256_bytes(system.encode())[:12]}")
    print("三组：BASE_RAW（原始Qwen+历史），V44_RAW（LoRA+历史），V44_CARDS（LoRA+历史+摘录卡片）")
    print("题目类型：", dict(Counter(q['category'] for q in questions)))
    print(f"有群聊上下文{sum(bool(q['context']) for q in questions)}题，没上下文{sum(not q['context'] for q in questions)}题（后者RAW和CARDS的提示相同）。")
    if args.check_only:
        check_duplicate_training(questions)
        for q in questions[:3]:
            print(f"\n----{q['id']} ----")
            print("原始版输入：", build_user_message(q, False)[:1200])
            print("卡片版输入：", build_user_message(q, True)[:1600])
        print("检查通过：不会向模型发送人工参考回复或评分标准；没有启动GPU、联网查询或QQ发送。")
        return
    if not (ADAPTER_DIR / "adapter_config.json").is_file():
        raise FileNotFoundError("缺少V4.4 LoRA adapter_config.json，请先确认训练已完成。")

    out = paths(args.tag)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if any(out[k].exists() for k in ("blind", "key")):
        raise FileExistsError("当前tag已有解盲或盲评文件，不允许覆盖；请换--tag")
    if args.resume:
        if not out["raw"].is_file() or not out["meta"].is_file():
            raise FileNotFoundError("--resume需要已有raw_long.csv和metadata.json")
        meta = json.loads(out["meta"].read_text(encoding="utf-8"))
        if meta.get("fingerprint") != fp:
            raise RuntimeError("续跑参数/测试题发生变化，不能复用先前结果。请使用原参数")
        old = read_csv(out["raw"])
        if len({(r["id"], r["repeat"], r["mode"]) for r in old}) != len(old):
            raise RuntimeError("raw_long.csv存在重复组合，取消续跑")
    else:
        if out["raw"].exists() or out["meta"].exists():
            raise FileExistsError(f"输出已存在：{out['raw']}；换--tag，或使用相同参数--resume")
        meta = {
            "status": "running", "fingerprint": fp, "base_model": BASE_MODEL,
            "adapter": str(ADAPTER_DIR), "data": str(DATA_FILE),
            "modes": list(MODES), "questions": len(questions), "repeats": args.repeats,
            "system_sha256": sha256_bytes(system.encode()),
            "total_generations": total,
            "generation": {"temperature": args.temperature, "top_p": 0.9,
                           "max_new_tokens": args.max_new_tokens,
                           "repetition_penalty": 1.15, "no_repeat_ngram_size": 4,
                           "enable_thinking": False, "max_seq_length": 1024},
            "disclosure": "CARDS group adds extractive structure; no new facts, long-term memory, reference answers, or web lookup",
        }
        out["meta"].write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        with out["raw"].open("x", newline="", encoding="utf-8-sig") as f:
            csv.DictWriter(f, RAW_FIELDS).writeheader()
        old = []
    done = {(r["id"], r["repeat"], r["mode"]) for r in old}
    print(f"开始生成：已有{len(done)}/{total}条，可随时Ctrl+C中断，按同参数加--resume继续。", flush=True)
    import unsloth  # noqa:F401 -- MUST load before torch/transformers/peft
    import torch
    from unsloth import FastModel
    from peft import PeftModel
    if not torch.cuda.is_available():
        raise RuntimeError("没有CUDA；请使用QWbot WSL + RTX 2070环境")
    print("GPU：", torch.cuda.get_device_name(0), flush=True)
    base, tokenizer = FastModel.from_pretrained(
        model_name=BASE_MODEL, max_seq_length=1024,
        dtype=torch.float16, load_in_4bit=True,
    )
    model = PeftModel.from_pretrained(base, str(ADAPTER_DIR), is_trainable=False)
    model.eval()
    if not hasattr(model, "disable_adapter") or not callable(model.disable_adapter):
        raise RuntimeError("当前PEFT缺少disable_adapter()，无法可靠对比原始底座。请升级PEFT/Unsloth，不要跳过。");
    eos_ids = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|im_end|>")]
    eos_ids = list(dict.fromkeys(v for v in eos_ids if isinstance(v, int) and v >= 0))
    if not eos_ids:
        raise RuntimeError("无法确定EOS token id")
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_ids[0]

    try:
        with out["raw"].open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, RAW_FIELDS)
            for q_index, q in enumerate(questions, 1):
                for repeat in range(1, args.repeats + 1):
                    seed = 3407 + q_index * 101 + repeat
                    for mode in MODES:
                        k = (q["id"], str(repeat), mode)
                        if k in done:
                            continue
                        text_input = build_user_message(q, mode == "V44_CARDS")
                        tok = tokenizer.apply_chat_template(
                            [{"role": "system", "content": system}, {"role": "user", "content": text_input}],
                            tokenize=True, add_generation_prompt=True, enable_thinking=False,
                            return_tensors="pt",
                        ).to("cuda")
                        n_tokens = tok.shape[-1]
                        if n_tokens + args.max_new_tokens > 1024:
                            raise RuntimeError(f"{q['id']} {mode}总tokens预计超过1024，不能静默截断；减少上下文或增加模型长度")
                        random.seed(seed)
                        torch.manual_seed(seed)
                        torch.cuda.manual_seed_all(seed)
                        ts = time.perf_counter()
                        context = model.disable_adapter() if mode == "BASE_RAW" else contextlib.nullcontext()
                        with context:
                            with torch.inference_mode():
                                generation = model.generate(
                                    input_ids=tok,
                                    max_new_tokens=args.max_new_tokens,
                                    do_sample=True, temperature=args.temperature, top_p=0.9,
                                    repetition_penalty=1.15, no_repeat_ngram_size=4,
                                    eos_token_id=eos_ids, pad_token_id=pad_id,
                                )
                        elapsed = round(time.perf_counter() - ts, 3)
                        out_ids = generation[0, n_tokens:]
                        reply = tokenizer.decode(out_ids, skip_special_tokens=True).strip()
                        record = {
                            "id": q["id"], "repeat": repeat, "mode": mode,
                            "category": q["category"], "priority": q["priority"],
                            "has_context": "y" if q["context"] else "n",
                            "seed": seed, "incoming_message": q["incoming_message"],
                            "context": json.dumps(q["context"], ensure_ascii=False),
                            "extracted_cards": json.dumps(make_context_cards(q), ensure_ascii=False) if mode == "V44_CARDS" else "",
                            "model_reply": reply, "time_seconds": elapsed,
                            "input_tokens": n_tokens, "output_tokens": len(out_ids),
                        }
                        writer.writerow(record)
                        f.flush()
                        done.add(k)
                        print(f"[{len(done)}/{total}] {q['id']} R{repeat} {mode} -> {reply} ({elapsed}s)", flush=True)
        rows = read_csv(out["raw"])
        expected = {(q["id"], str(rep), mode)
                    for q in questions for rep in range(1, args.repeats + 1) for mode in MODES}
        seen = {(r["id"], r["repeat"], r["mode"]) for r in rows}
        if seen != expected or len(rows) != len(expected):
            raise RuntimeError(f"不完整或重复：已有{len(rows)}条，预期{len(expected)}条；请--resume")
        by_key = {(r["id"], r["repeat"], r["mode"]): r for r in rows}
        key = []
        blind_rows = []
        for q in questions:
            for repeat in range(1, args.repeats + 1):
                code = f"{q['id']}-R{repeat}"
                shuffled = list(MODES)
                random.Random(seed_from_id(code)).shuffle(shuffled)
                mapping = dict(zip("ABC", shuffled))
                key.append({"id": code, **mapping})
                blind_row = {
                    "id": q["id"], "repeat": repeat,
                    "category": q["category"], "priority": q["priority"],
                    "has_context": "y" if q["context"] else "n",
                    "context": json.dumps(q["context"], ensure_ascii=False),
                    "incoming_message": q["incoming_message"],
                    "evaluation_criteria": q["evaluation_criteria"],
                    **{f"reply_{a}": by_key[(q["id"], str(repeat), mode)]["model_reply"]
                       for a, mode in mapping.items()},
                }
                blind_rows.append(blind_row)
        with out["blind"].open("x", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, BLIND_FIELDS)
            writer.writeheader()
            writer.writerows(blind_rows)
        out["key"].write_text(json.dumps({
            "purpose": "盲评前不要查看：A/B/C的真实身份映射", "mapping": key,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        meta["status"] = "completed"
        meta["completed_generations"] = len(rows)
        meta["blind_rows"] = len(blind_rows)
        out["meta"].write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        print("\n三组推理完成。仅需上传匿名评分文件：", out["blind"])
        print("先不要上传/打开解盲文件：", out["key"])
        print("原始逐条输出仅供调试：", out["raw"])
    except BaseException:
        print(f"生成中断，原始结果已保存到{out['raw']}；用原参数+--resume继续。", file=sys.stderr)
        raise


def seed_from_id(key: str) -> int:
    return int.from_bytes(hashlib.sha256(("QWbot-triad-v1:" + key).encode()).digest()[:8], "big")


if __name__ == "__main__":
    main()
