#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""同一组提示词，对比原始Qwen、抗吧V2/V3/V4的回复质量。

使用：python compare_kangba_v4.py --limit 8
      python compare_kangba_v4.py --limit 0    # 全部测试问题
只读取模型，不训练、不覆盖任何LoRA。
"""
import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", str(ROOT / "hf_cache"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import unsloth  # noqa: F401 -- MUST import before torch and peft
import torch
from peft import PeftModel
from unsloth import FastModel

MODEL = "unsloth/Qwen3-1.7B-unsloth-bnb-4bit"
ADAPTERS = {
    "抗吧V2": ("kangba_v2", ROOT / "outputs" / "kangba_v2_lora"),
    "抗吧V3": ("kangba_v3", ROOT / "outputs" / "kangba_v3_lora"),
    "抗吧V4": ("kangba_v4", ROOT / "outputs" / "kangba_v4_lora"),
}
SYSTEM = (
    "你是QQ群里的普通群友，说话带抗压背锅吧那种简短、抽象、反串、回旋镖的味道。"
    "先理解对方在说什么，接得上话才整活；不用客服腔，不长篇说教，不硬塞游戏梗。"
    "抗吧常聊电竞但不限电竞，日常水帖也能正常聊；真实事件不编造，严肃求助时正常回应。"
)
HOLDOUT_FILE = ROOT / "datasets" / "kangba_v4_kangba_holdout_prompts.txt"
DEFAULT_TESTS = [
    "我上单0/8，这把全怪打野",
    "BLG今年绝对夺冠",
    "昨天说绝不熬夜，今天又通宵了",
    "你们抗抗就知道串",
    "我宣布这个新队已经是冠军了",
    "今天真的有点累，不想说话",
    "这个梗最早是从哪来的",
    "我刚被老板骂了一顿，今天不想上班",
    "别人赢了就说版本好，我们赢了就是硬实力",
    "谁说退坑了？刚刚还看见他在线",
    "我一张嘴全群就开始吵架",
    "下个月还没到，我已经宣布自己是销冠了",
]


def questions():
    if HOLDOUT_FILE.exists():
        fromfile = [s.strip() for s in HOLDOUT_FILE.read_text(encoding="utf-8").splitlines() if s.strip()]
        if fromfile:
            return list(dict.fromkeys(fromfile + DEFAULT_TESTS[7:]))
    return DEFAULT_TESTS


def main():
    argp = argparse.ArgumentParser(description="原始Qwen / V2 / V3 / V4 同条件抗吧聊天对比")
    argp.add_argument("--limit", type=int, default=8, help="默认8条，0表示全部")
    argp.add_argument("--max-new-tokens", type=int, default=48)
    args = argp.parse_args()
    if args.max_new_tokens <= 0:
        raise ValueError("max-new-tokens必须>0")
    available = [(label, name, path) for label, (name, path) in ADAPTERS.items()
                 if (path / "adapter_config.json").exists()]
    if not any(label == "抗吧V4" for label, _, _ in available):
        raise FileNotFoundError("找不到 outputs/kangba_v4_lora/adapter_config.json；请先训练V4")
    missing = [label for label in ADAPTERS if label not in {x[0] for x in available}]
    if missing:
        print("未找到适配器，自动跳过：", ", ".join(missing), flush=True)

    print("加载原始Qwen3 1.7B与LoRA适配器……", flush=True)
    model, tokenizer = FastModel.from_pretrained(
        model_name=MODEL, max_seq_length=512,
        dtype=torch.float16, load_in_4bit=True,
    )
    first_label, first_name, first_path = available[0]
    model = PeftModel.from_pretrained(model, str(first_path), adapter_name=first_name)
    for _, name, path in available[1:]:
        model.load_adapter(str(path), adapter_name=name)
    model.eval()

    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    eos = tokenizer.eos_token_id
    eos_ids = list(dict.fromkeys(x for x in (eos, im_end) if isinstance(x, int) and x >= 0))
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos

    def answer_one(text, mode, seed):
        message = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}]
        ids = tokenizer.apply_chat_template(
            message, tokenize=True, add_generation_prompt=True,
            enable_thinking=False, return_tensors="pt"
        ).to("cuda")
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        kwargs = dict(
            input_ids=ids,
            # Avoid max_new_tokens/max_length warning caused by the model default max_length=20.
            max_length=ids.shape[-1] + args.max_new_tokens,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            repetition_penalty=1.15,
            no_repeat_ngram_size=4,
            pad_token_id=pad_id,
            eos_token_id=eos_ids,
        )
        with torch.inference_mode():
            if mode == "base":
                with model.disable_adapter():
                    generated = model.generate(**kwargs)
            else:
                model.set_adapter(mode)
                generated = model.generate(**kwargs)
        text = tokenizer.decode(generated[0, ids.shape[-1]:], skip_special_tokens=True).strip()
        return text or "（未生成可见回复）"

    items = questions()
    if args.limit > 0:
        items = items[:args.limit]
    rows, lines = [], []
    modes = [("原始Qwen", "base")] + [(label, name) for label, name, _ in available]
    for i, prompt in enumerate(items, 1):
        print(f"\n========== 测试 {i}/{len(items)} ==========", flush=True)
        print("群友：", prompt, flush=True)
        result = {"id": i, "prompt": prompt, "outputs": {}}
        lines.extend([f"========== 测试 {i}/{len(items)} ==========", f"群友：{prompt}"])
        for display, mode in modes:
            answer = answer_one(prompt, mode, 3407 + i)
            result["outputs"][display] = answer
            print(f"{display}：{answer}", flush=True)
            lines.append(f"{display}：{answer}")
        lines.append("")
        rows.append(result)
    out = ROOT / "outputs"
    out.mkdir(exist_ok=True)
    txt = out / "kangba_v4_comparison.txt"
    js = out / "kangba_v4_comparison.json"
    txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    js.write_text(json.dumps({"system_prompt": SYSTEM, "generation": {
        "temperature": 0.8, "top_p": 0.9, "repetition_penalty": 1.15,
        "max_new_tokens": args.max_new_tokens}, "tests": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n完成。文本报告:", txt, "\n结构化报告:", js, flush=True)
    print("本测试沿用一致的V4系统提示词，适配器之间的效果仅作定性对照。", flush=True)


if __name__ == "__main__":
    main()
