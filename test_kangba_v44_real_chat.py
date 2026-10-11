#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""QWbot V4.4: run anonymized real-chat shadow tests (offline, no QQ sending, no training).

python test_kangba_v44_real_chat.py --check-only
python test_kangba_v44_real_chat.py --limit 5 --tag quick
python test_kangba_v44_real_chat.py --limit 0 --repeats 2 --tag full
"""
from __future__ import annotations
import argparse
import csv
import json
import os
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", str(ROOT / "hf_cache"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
MODEL_NAME = "unsloth/Qwen3-1.7B-unsloth-bnb-4bit"
ADAPTER = ROOT / "outputs/kangba_v44_lora"
DATA_FILE = ROOT / "datasets/QWbot_V44_真实群聊影子测试_匿名31题.jsonl"
TRAIN_FILE = ROOT / "datasets/kangba_v44_train.jsonl"
OUTPUT_DIR = ROOT / "outputs"

FALLBACK_SYSTEM = (
    "你是QQ群里长期混抗压背锅吧的普通群友，不是客服。先听懂对方在说什么，再自然接话。"
    "可以用反串、回旋镖、阴阳怪气，也可以自然使用“典”“绷”“闹麻了”等短梗；该用才用，"
    "不要随机拼接黑话、硬套游戏话题或无意义复读。一般一句话就够，不强制字数；"
    "认真求助和明确不想被开玩笑时正常回应。新闻、赛事、转会等事实未知时，"
    "不编造、不假装已查过。只输出要发到QQ群的回复。"
)
FIELDS = [
    "id", "repeat", "category", "priority", "context", "incoming_message", "model_reply",
    "evaluation_criteria", "example_reply_for_human_review", "seed", "time_seconds",
    "understanding_0_2", "style_0_2", "naturalness_0_2", "fabrication_or_harm_y_n", "notes",
]


def read_dataset(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"测试题未放到指定位置：{path}")
    data = []
    with path.open(encoding="utf-8-sig") as stream:
        for index, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"测试题第{index}行不是有效JSON：{exc}") from exc
            required = ["id", "category", "priority", "context", "incoming_message", "evaluation_criteria"]
            if not all(key in item for key in required):
                raise ValueError(f"测试题第{index}行缺少必需字段")
            if not isinstance(item["context"], list) or not all(
                isinstance(m, dict) and isinstance(m.get("speaker"), str)
                and isinstance(m.get("text"), str) for m in item["context"]
            ):
                raise ValueError(f"测试题第{index}行 context 格式错误")
            if not (isinstance(item["incoming_message"], str) and item["incoming_message"].strip()):
                raise ValueError(f"测试题第{index}行没有问题文本")
            data.append(item)
    if not data:
        raise ValueError("测试题为空")
    if len(set(row["id"] for row in data)) != len(data):
        raise ValueError("测试题ID重复")
    return data


def build_user_message(item: dict) -> str:
    """One structured user turn, matching Qwen V4.4 system+user format.

    The historical statements are quoted as context, NOT instructions or model-proven facts.
    """
    messages = item["context"]
    if messages:
        history = "\n".join(f"{m['speaker']}：{m['text']}" for m in messages)
        return (f"以下是群里的聊天记录（只是上下文）：\n{history}\n\n"
                f"当前群友发言：{item['incoming_message']}\n请只回复当前群友。")
    return item["incoming_message"]


def read_system() -> str:
    if TRAIN_FILE.exists():
        with TRAIN_FILE.open(encoding="utf-8-sig") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                messages = item.get("prompt", [])
                if messages and messages[0].get("role") == "system":
                    return messages[0]["content"]
                break
    print("提示：没有找到训练集系统提示词，使用V4.4同版的内置系统提示词。")
    return FALLBACK_SYSTEM


def validate_args(args):
    if args.limit < 0 or args.repeats not in range(1, 6):
        raise ValueError("--limit >= 0，--repeats 在1到5之间")
    if not re.fullmatch(r"[a-zA-Z0-9_-]{1,40}", args.tag):
        raise ValueError("--tag 仅允许英文、数字、下划线和短横线")
    if args.max_new_tokens < 8 or args.max_new_tokens > 256:
        raise ValueError("--max-new-tokens 在8到256之间")
    if not 0 < args.temperature <= 2:
        raise ValueError("--temperature 在(0,2]之间")


def main():
    cli = argparse.ArgumentParser(description="V4.4模型：31道匿名真实QQ群影子测试，不发QQ群，不修改LoRA")
    cli.add_argument("--check-only", action="store_true", help="只检查数据，不加载模型，不占用GPU")
    cli.add_argument("--limit", type=int, default=0, help="只测前N道题；0为全部31题")
    cli.add_argument("--repeats", type=int, default=1, help="每题生成次数；建议先1次再2次")
    cli.add_argument("--tag", default="real31", help="输出文件名标签，可用quick、full等避免覆盖")
    cli.add_argument("--max-new-tokens", type=int, default=80)
    cli.add_argument("--temperature", type=float, default=0.8)
    args = cli.parse_args()
    validate_args(args)

    dataset = read_dataset(DATA_FILE)
    questions = dataset[:args.limit] if args.limit else dataset
    system = read_system()
    print(f"测试题: {len(questions)}/{len(dataset)}；每题生成 {args.repeats} 次；总生成 {len(questions)*args.repeats} 次")
    print("分类：", dict(Counter(item['category'] for item in questions)))
    print("LoRA：", ADAPTER)
    print("提示词：", system)
    print("第一题实际送入模型的user消息：\n", build_user_message(questions[0]))
    print("提示：评价标准和人工参考答案不会发送给模型。")
    if args.check_only:
        print("检查通过：未加载GPU模型；未训练；未发送QQ群消息；未生成结果文件。")
        return

    if not (ADAPTER / "adapter_config.json").is_file():
        raise FileNotFoundError(f"找不到V4.4 LoRA：{ADAPTER}（需要adapter_config.json）")
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    output_csv = OUTPUT_DIR / f"kangba_v44_real_chat_{args.tag}_scoring.csv"
    output_meta = OUTPUT_DIR / f"kangba_v44_real_chat_{args.tag}_metadata.json"
    if output_csv.exists() or output_meta.exists():
        raise FileExistsError(f"为避免覆盖，已有同名结果：{output_csv}；请换--tag或备份后删除旧文件")

    # Intentionally import Unsloth before Torch, PEFT and Transformers.
    import unsloth  # noqa: F401
    import torch
    from unsloth import FastModel
    from peft import PeftModel

    if not torch.cuda.is_available():
        raise RuntimeError("没有检测到CUDA GPU，请在WSL QWbot环境中运行")
    print("GPU:", torch.cuda.get_device_name(0), flush=True)
    print("正在加载Qwen3-1.7B 4bit以及V4.4 LoRA……", flush=True)
    base, tokenizer = FastModel.from_pretrained(
        model_name=MODEL_NAME, max_seq_length=512,
        dtype=torch.float16, load_in_4bit=True,
    )
    model = PeftModel.from_pretrained(base, str(ADAPTER), is_trainable=False)
    model.eval()
    eos = tokenizer.eos_token_id
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    eos_ids = list(dict.fromkeys(v for v in [eos, im_end] if isinstance(v, int) and v >= 0))
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_ids[0]

    # Check final output path at the moment of file creation (mode x never overwrites).
    completed = 0
    try:
        with output_csv.open("x", newline="", encoding="utf-8-sig") as fout:
            writer = csv.DictWriter(fout, fieldnames=FIELDS)
            writer.writeheader()
            for idx, item in enumerate(questions, 1):
                user_text = build_user_message(item)
                inputs = tokenizer.apply_chat_template(
                    [{"role": "system", "content": system}, {"role": "user", "content": user_text}],
                    tokenize=True, add_generation_prompt=True, enable_thinking=False,
                    return_tensors="pt",
                ).to("cuda")
                n_tokens = inputs.shape[-1]
                if n_tokens > 480:
                    print(f"警告：{item['id']} 提示词有{n_tokens}tokens，可能较长")
                for rep in range(1, args.repeats + 1):
                    seed = 3407 + idx * 101 + rep
                    random.seed(seed)
                    torch.manual_seed(seed)
                    torch.cuda.manual_seed_all(seed)
                    t0 = time.perf_counter()
                    with torch.inference_mode():
                        generated = model.generate(
                            input_ids=inputs,
                            max_new_tokens=args.max_new_tokens,
                            do_sample=True,
                            temperature=args.temperature,
                            top_p=0.9,
                            repetition_penalty=1.15,
                            no_repeat_ngram_size=4,
                            eos_token_id=eos_ids,
                            pad_token_id=pad_id,
                        )
                    elapsed = round(time.perf_counter() - t0, 2)
                    reply = tokenizer.decode(generated[0, n_tokens:], skip_special_tokens=True).strip()
                    row = {
                        "id": item["id"], "repeat": rep,
                        "category": item["category"], "priority": item["priority"],
                        "context": json.dumps(item["context"], ensure_ascii=False),
                        "incoming_message": item["incoming_message"], "model_reply": reply,
                        "evaluation_criteria": item["evaluation_criteria"],
                        "example_reply_for_human_review": item.get("example_reply_for_human_review", ""),
                        "seed": seed, "time_seconds": elapsed,
                        "understanding_0_2": "", "style_0_2": "", "naturalness_0_2": "",
                        "fabrication_or_harm_y_n": "", "notes": "",
                    }
                    writer.writerow(row)
                    fout.flush()  # keeps already generated rows on interruption
                    completed += 1
                    print(f"[{completed}/{len(questions)*args.repeats}] {item['id']}-R{rep} "
                          f"群友: {item['incoming_message']}\n  V4.4: {reply}（{elapsed}s）", flush=True)
        output_meta.write_text(json.dumps({
            "model": MODEL_NAME,
            "adapter": str(ADAPTER),
            "dataset": str(DATA_FILE),
            "count_questions": len(questions), "repeats": args.repeats,
            "total_rows": completed, "test_type": "anonymized-offline-shadow",
            "generation": {"enable_thinking": False, "temperature": args.temperature,
                           "top_p": 0.9, "max_new_tokens": args.max_new_tokens,
                           "repetition_penalty": 1.15, "no_repeat_ngram_size": 4},
            "note": "context & incoming_message passed to model; criteria and reference excluded from input",
        }, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        print(f"已生成 {completed} 条可在文件中恢复：{output_csv}", file=sys.stderr)
        raise
    print("\n全部完成！请上传此文件供人工评分：", output_csv)
    print("测试参数记录：", output_meta)
    print("注意：这不是实时NapCat监听；没有向QQ群发消息；模型未重新训练。")


if __name__ == "__main__":
    main()
