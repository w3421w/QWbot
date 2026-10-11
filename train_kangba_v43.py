#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""QWbot V4.3: 抗吧风格（不限LOL）的Qwen3-1.7B QLoRA训练。

在 ~/Qwen_Finetune 执行：
  python prepare_kangba_v43.py build
  python train_kangba_v43.py --check-only
  python train_kangba_v43.py

注意：从原始Qwen加载，不继承V2/V3/V4；不改动其他LoRA。
"""
import argparse
import json
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", str(ROOT / "hf_cache"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

BASE_MODEL = "unsloth/Qwen3-1.7B-unsloth-bnb-4bit"
TRAIN_FILE = ROOT / "datasets" / "kangba_v43_train.jsonl"
VALID_FILE = ROOT / "datasets" / "kangba_v43_valid.jsonl"
OUTPUT_DIR = ROOT / "outputs" / "kangba_v43_lora"
CHECKPOINT_DIR = ROOT / "outputs" / "kangba_v43_checkpoints"
DEFAULT_MAX_LENGTH = 512
SEED = 3407


def normalized(s):
    return re.sub(r"\s+", "", str(s)).casefold()


def read_rows(path):
    if not path.exists():
        raise FileNotFoundError(f"找不到文件：{path}\n请先运行 python prepare_kangba_v43.py build")
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for n, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path.name} 第{n}行JSON格式错误：{exc}") from exc
            prompt, completion = item.get("prompt"), item.get("completion")
            if (not isinstance(prompt, list) or not prompt or not isinstance(completion, list)
                or len(completion) != 1 or not isinstance(completion[0], dict)
                or completion[0].get("role") != "assistant"
                or not isinstance(completion[0].get("content"), str)
                or not completion[0]["content"].strip()
                or not all(isinstance(m, dict) and m.get("role") in {"system", "user", "assistant"}
                           and isinstance(m.get("content"), str) for m in prompt)
                or not any(m["role"] == "user" for m in prompt)):
                raise ValueError(f"{path.name} 第{n}行不是预期的prompt/completion对话")
            if len(prompt) != 2 or [x["role"] for x in prompt] != ["system", "user"]:
                raise ValueError(f"{path.name} 第{n}行不是本轮V4.3的system+user对话格式")
            rows.append((prompt, completion[0]["content"].strip()))
    if not rows:
        raise ValueError(f"{path.name} 为空")
    return rows


def check_data(train, valid):
    ptrain = {normalized(msgs[-1]["content"]) for msgs, _ in train}
    atrain = {normalized(ans) for _, ans in train}
    pvalid = {normalized(msgs[-1]["content"]) for msgs, _ in valid}
    avalid = {normalized(ans) for _, ans in valid}
    if ptrain & pvalid or atrain & avalid:
        raise ValueError("训练/验证集有相同提问或答案，请重新构建并检查划分")
    systems = {msgs[0]["content"] for msgs, _ in (train + valid)}
    if len(systems) != 1:
        raise ValueError(f"发现{len(systems)}种system提示词；本次实验要求训练模板统一")
    out = {
        "train_count": len(train), "valid_count": len(valid),
        "system_prompt": next(iter(systems)),
        "train_avg_reply_chars": round(sum(len(a) for _, a in train) / len(train), 1),
        "valid_avg_reply_chars": round(sum(len(a) for _, a in valid) / len(valid), 1),
    }
    return out


def build_hf_dataset(chat_rows, tokenizer, max_length, split):
    from datasets import Dataset
    out, skipped = [], 0
    for messages, answer in chat_rows:
        prefix = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        completion = answer + "<|im_end|>"
        prompt_tokens = tokenizer(prefix, add_special_tokens=False)["input_ids"]
        joined_tokens = tokenizer(prefix + completion, add_special_tokens=False)["input_ids"]
        if len(joined_tokens) > max_length or len(prompt_tokens) >= max_length - 4:
            skipped += 1
            continue
        out.append({"prompt": prefix, "completion": completion})
    if not out:
        raise ValueError(f"{split}全部超出max_length={max_length}，需增大长度或精简语料")
    print(f"{split}: {len(out)}条可训练，{skipped}条因过长跳过")
    return Dataset.from_list(out), skipped


def main():
    cli = argparse.ArgumentParser(description="QWbot抗吧风格V4.3 Qwen3 QLoRA")
    cli.add_argument("--check-only", action="store_true", help="不加载GPU模型，仅验证V4数据")
    cli.add_argument("--epochs", type=float, default=3.0, help="默认3轮；小数据先做对照实验")
    cli.add_argument("--lr", type=float, default=5e-5, help="默认5e-5，比V3略高")
    cli.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    cli.add_argument("--overwrite", action="store_true", help="显式允许覆盖已存在的V4输出")
    args = cli.parse_args()
    train_rows, valid_rows = read_rows(TRAIN_FILE), read_rows(VALID_FILE)
    summary = check_data(train_rows, valid_rows)
    print("V4.3数据检查通过:")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.check_only:
        print("仅检查完成，未加载模型、未启动训练。")
        return
    if args.epochs <= 0 or args.lr <= 0 or args.max_length < 128:
        raise ValueError("训练轮数、学习率须大于0，max-length须至少128")
    if (OUTPUT_DIR / "adapter_config.json").exists() and not args.overwrite:
        raise FileExistsError(f"V4.3 LoRA已经存在：{OUTPUT_DIR}\n如需重训，请先备份，再使用 --overwrite")

    # 保证Unsloth先于torch、transformers、peft、trl导入。
    import unsloth  # noqa: F401
    import torch
    from unsloth import FastModel
    from trl import SFTConfig, SFTTrainer

    if not torch.cuda.is_available():
        raise RuntimeError("未检测到CUDA；请在可访问RTX2070的QWbot WSL环境运行")
    torch.manual_seed(SEED)
    print("GPU:", torch.cuda.get_device_name(0))
    print("正在从原始Qwen3 1.7B 4bit加载（不继承V2/V3 LoRA）...")
    model, tokenizer = FastModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=args.max_length,
        dtype=torch.float16,
        load_in_4bit=True,
    )
    model = FastModel.get_peft_model(
        model,
        r=8,
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth",
        random_state=SEED,
    )
    train_dataset, skipped_train = build_hf_dataset(train_rows, tokenizer, args.max_length, "训练集")
    valid_dataset, skipped_valid = build_hf_dataset(valid_rows, tokenizer, args.max_length, "验证集")

    config = SFTConfig(
        output_dir=str(CHECKPOINT_DIR),
        max_length=args.max_length,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=1,
        learning_rate=args.lr,
        warmup_ratio=0.05,
        lr_scheduler_type="linear",
        fp16=True,
        bf16=False,
        optim="adamw_torch",
        gradient_checkpointing=True,
        max_grad_norm=1.0,
        logging_steps=5,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        packing=False,
        completion_only_loss=True,
        assistant_only_loss=False,
        seed=SEED,
        data_seed=SEED,
    )
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        args=config,
        processing_class=tokenizer,
    )

    # 检验训练标签：系统/用户输入屏蔽，assistant回复才参与损失。
    sample = trainer.data_collator([trainer.train_dataset[0]])
    input_ids = sample["input_ids"][0].tolist()
    labels = sample["labels"][0].tolist()
    supervised = [i for i, y in zip(input_ids, labels) if y != -100]
    masked = sum(1 for y in labels if y == -100)
    preview = tokenizer.decode(supervised, skip_special_tokens=False)
    expected = train_dataset[0]["completion"].removesuffix("<|im_end|>")
    print(f"标签检查：prompt屏蔽{masked} tokens；回复监督{len(supervised)} tokens")
    print("目标助手回复预览:", repr(preview[:160]))
    if masked <= 0 or not supervised or expected[:min(4, len(expected))] not in preview:
        raise RuntimeError("助手回复标签检查未通过，已取消训练，避免学习错误上下文")
    if "<|im_start|>user" in preview or "<|im_start|>system" in preview:
        raise RuntimeError("监督标签中出现了用户/系统输入，已取消训练")
    print(f"开始V4：epochs={args.epochs}, learning_rate={args.lr}, max_length={args.max_length}")
    result = trainer.train()
    validation = trainer.evaluate()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    report = {
        "base_model": BASE_MODEL,
        "adapter_path": str(OUTPUT_DIR),
        "train_count": len(train_dataset),
        "valid_count": len(valid_dataset),
        "skipped_train": skipped_train,
        "skipped_valid": skipped_valid,
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "max_length": args.max_length,
        "lora_rank": 8,
        "lora_alpha": 16,
        "train_loss": result.training_loss,
        "eval_loss": validation.get("eval_loss"),
        "best_model_checkpoint": trainer.state.best_model_checkpoint,
        "peak_cuda_allocated_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
        "log_history": trainer.state.log_history,
        "system_prompt": summary["system_prompt"],
    }
    report_path = OUTPUT_DIR / "training_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n========= 抗吧V4.3训练完成 =========")
    print("训练Loss:", report["train_loss"])
    print("验证Loss:", report["eval_loss"])
    print("最佳轮次checkpoint:", report["best_model_checkpoint"])
    print("PyTorch显存峰值:", report["peak_cuda_allocated_gb"], "GB")
    print("V4.3 LoRA:", OUTPUT_DIR)
    print("报告:", report_path)
    print("提醒：验证集不足20条，Loss仅供参考；必须比较真实对话效果。")


if __name__ == "__main__":
    main()
