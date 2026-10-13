# -*- coding: utf-8 -*-
"""Compare base Qwen3-1.7B against Kangba V2 / V3 LoRA adapters.

Run in ~/Qwen_Finetune with the existing WSL venv:
    python compare_kangba_v3.py

This script does not train or alter model weights.
"""
import os
from pathlib import Path

# These must be set before importing Unsloth / Hugging Face libraries.
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", str(Path.home() / "Qwen_Finetune" / "hf_cache"))

import unsloth  # noqa: F401; patch BEFORE torch, transformers and peft
import torch
from unsloth import FastModel
from peft import PeftModel

BASE = "unsloth/Qwen3-1.7B-unsloth-bnb-4bit"
V2 = Path("outputs/kangba_v2_lora")
V3 = Path("outputs/kangba_v3_lora")
REPORT = Path("outputs/kangba_v3_comparison.txt")
MAX_NEW_TOKENS = 48

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
SYSTEM = (
    "你是QQ群里的AI群友，风格偏抗压背锅吧。"
    "回复自然简短，擅长反串和接梗。"
    "不知道的最新赛事和转会不乱编，别人难受时正常关心。"
)


def main():
    for adapter in (V2, V3):
        if not (adapter / "adapter_config.json").exists():
            raise FileNotFoundError(f"找不到LoRA适配器：{adapter.resolve()}")

    print("加载基础Qwen3 1.7B及V2、V3 LoRA……", flush=True)
    model, tokenizer = FastModel.from_pretrained(
        model_name=BASE,
        max_seq_length=512,
        dtype=torch.float16,
        load_in_4bit=True,
    )
    model = PeftModel.from_pretrained(model, str(V2), adapter_name="kangba_v2")
    model.load_adapter(str(V3), adapter_name="kangba_v3")
    model.eval()

    def respond(question, kind, seed):
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": question},
        ]
        input_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_tensors="pt",
        ).to("cuda")

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Use max_length = input tokens + new tokens to avoid the
        # max_length / max_new_tokens precedence warning.
        generation_kwargs = dict(
            input_ids=input_ids,
            max_length=input_ids.shape[-1] + MAX_NEW_TOKENS,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            repetition_penalty=1.15,
            no_repeat_ngram_size=4,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        with torch.inference_mode():
            if kind == "original":
                with model.disable_adapter():
                    output = model.generate(**generation_kwargs)
            else:
                model.set_adapter(kind)
                output = model.generate(**generation_kwargs)
        answer = tokenizer.decode(
            output[0, input_ids.shape[-1]:],
            skip_special_tokens=True,
        ).strip()
        return answer if answer else "（未生成可见文本）"

    lines = []
    for idx, question in enumerate(QUESTIONS, 1):
        lines.append(f"========== 测试 {idx}/{len(QUESTIONS)} ==========")
        lines.append(f"群友：{question}")
        for kind, display in (
            ("original", "原始Qwen"),
            ("kangba_v2", "抗吧V2"),
            ("kangba_v3", "抗吧V3"),
        ):
            answer = respond(question, kind, seed=3407 + idx)
            lines.append(f"{display}：{answer}")
        lines.append("")
        print("\n".join(lines[-5:]), flush=True)

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n已保存对比结果：{REPORT.resolve()}", flush=True)
    print("注意：此测试比较回答表现，不代表验证集统计显著性。", flush=True)


if __name__ == "__main__":
    main()
