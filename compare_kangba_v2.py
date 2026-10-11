#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Qwen3 1.7B 原版 vs 抗吧 V2 LoRA：防重复生成对比测试。

运行：
    cd ~/Qwen_Finetune
    source .venv/bin/activate
    export HF_HOME="$HOME/Qwen_Finetune/hf_cache"
    python compare_kangba_v2.py

仅用于比较推理行为，不会继续训练或修改 LoRA。
"""

import os
import random
import warnings
from contextlib import nullcontext
from pathlib import Path

# 在 import huggingface_hub / unsloth 之前设置环境变量。
BASE_DIR = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(BASE_DIR / "hf_cache"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from peft import PeftModel
from unsloth import FastModel

MODEL_NAME = "unsloth/Qwen3-1.7B-unsloth-bnb-4bit"
ADAPTER_PATH = BASE_DIR / "outputs" / "kangba_v2_lora"
OUTPUT_PATH = BASE_DIR / "outputs" / "kangba_v2_comparison.txt"

MAX_CONTEXT = 512
MAX_NEW_TOKENS = 48
SEED = 3407

SYSTEM_PROMPT = (
    "你是QQ群里的AI群友，风格偏抗压背锅吧。"
    "日常一句话回复，简短自然，擅长反串、阴阳怪气和接梗。"
    "对赛果、转会等未经核实的事实不乱编，别人明显疲惫时正常回应。"
)

QUESTIONS = [
    "我上单0/8，这把全怪打野",
    "BLG今年绝对夺冠",
    "这选手赛前吹上天，赛后直接隐身",
    "我LOL已经十连败了",
    "我瓦洛兰特队友天天干拉送人头",
    "我们队才赢第一局，冠军已经稳了",
    "这个选手是不是昨天转会了？",
    "今天不想打游戏了，感觉好累",
]


def generate_text(model, tokenizer, question, use_lora):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_tensors="pt",
    )
    encoded = encoded.to("cuda")

    # 两种模式使用相同随机种子，使对比更公平。
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    # 推理参数用于测试减少重复；这不会修复训练数据质量问题。
    generation_kwargs = dict(
        input_ids=encoded,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=True,
        temperature=0.8,
        top_p=0.9,
        repetition_penalty=1.15,
        no_repeat_ngram_size=4,
        pad_token_id=tokenizer.eos_token_id,
    )

    # 当前安装的 transformers 版本可能在模型 generation_config 中自带
    # max_length=40960，再传入 max_new_tokens 会给出无害提示。
    # 使用一个副本恢复默认 max_length，避免该冗余提示。
    generation_config = model.generation_config
    original_max_length = getattr(generation_config, "max_length", None)
    if original_max_length is not None:
        generation_config.max_length = 20

    context = nullcontext() if use_lora else model.disable_adapter()
    try:
        with context, torch.inference_mode():
            output = model.generate(**generation_kwargs)
    finally:
        if original_max_length is not None:
            generation_config.max_length = original_max_length

    decoded = tokenizer.decode(
        output[0][encoded.shape[-1]:],
        skip_special_tokens=True,
    ).strip()
    return decoded or "（无输出）"


def main():
    if not ADAPTER_PATH.joinpath("adapter_config.json").is_file():
        raise FileNotFoundError(
            f"找不到 LoRA 文件：{ADAPTER_PATH / 'adapter_config.json'}\n"
            "请先确认你已完成 V2 微调。"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA不可用，请先用 python -c 'import torch; print(torch.cuda.is_available())' 检查。")

    print("正在加载原始 Qwen3 1.7B（4-bit）与 V2 LoRA……", flush=True)
    model, tokenizer = FastModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_CONTEXT,
        dtype=torch.float16,
        load_in_4bit=True,
    )
    model = PeftModel.from_pretrained(model, str(ADAPTER_PATH))
    model.eval()

    print("GPU:", torch.cuda.get_device_name(0), flush=True)
    print("采样参数: temperature=0.8, top_p=0.9, repetition_penalty=1.15, no_repeat_ngram_size=4")

    lines = [
        "Qwen3 1.7B：原始模型 vs 抗吧V2 LoRA（防重复采样测试）",
        f"模型: {MODEL_NAME}",
        f"LoRA: {ADAPTER_PATH}",
        "参数: max_new_tokens=48, do_sample=True, temperature=0.8, "
        "top_p=0.9, repetition_penalty=1.15, no_repeat_ngram_size=4, seed=3407",
        "",
    ]
    for index, question in enumerate(QUESTIONS, 1):
        print(f"\n========== 测试 {index}/{len(QUESTIONS)} ==========", flush=True)
        print("群友：", question, flush=True)
        original = generate_text(model, tokenizer, question, use_lora=False)
        tuned = generate_text(model, tokenizer, question, use_lora=True)
        print("原始Qwen：", original, flush=True)
        print("抗吧V2：", tuned, flush=True)

        lines.extend([
            f"========== 测试 {index} ==========",
            f"群友：{question}",
            f"原始Qwen：{original}",
            f"抗吧V2：{tuned}",
            "",
        ])

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n全部测试完成，结果保存至：{OUTPUT_PATH}", flush=True)
    print("注意：如果LoRA仍明显复读，下一步要检查数据质量与训练模板，而不是继续提高防重复惩罚。")


if __name__ == "__main__":
    main()
