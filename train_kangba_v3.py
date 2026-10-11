#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Qwen3-1.7B / Unsloth: short-response style QLoRA V3.

Input:
  datasets/kangba_style_train.jsonl (75 chat prompt/completion records)
  datasets/kangba_style_valid.jsonl (7 chat prompt/completion records)

Training is deliberately initialized from the ORIGINAL Qwen3 model, not V2.
Training uses Qwen3 non-thinking prompt format and masks prompt tokens from loss.
"""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(ROOT / "hf_cache"))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Unsloth must be loaded before torch/transformers/peft/trl.
import unsloth  # noqa: F401

import json
import torch
from datasets import Dataset
from trl import SFTConfig, SFTTrainer
from unsloth import FastModel

MODEL_NAME = "unsloth/Qwen3-1.7B-unsloth-bnb-4bit"
DATA_DIR = ROOT / "datasets"
TRAIN_FILE = DATA_DIR / "kangba_style_train.jsonl"
VALID_FILE = DATA_DIR / "kangba_style_valid.jsonl"
OUTPUT_DIR = ROOT / "outputs" / "kangba_v3_lora"
CHECKPOINT_DIR = ROOT / "outputs" / "kangba_v3_checkpoints"
MAX_LENGTH = 512
SEED = 3407


def load_chat_rows(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"找不到数据文件: {path}")
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            prompt, completion = row.get("prompt"), row.get("completion")
            if (not isinstance(prompt, list) or not isinstance(completion, list)
                    or len(completion) != 1
                    or completion[0].get("role") != "assistant"
                    or not isinstance(completion[0].get("content"), str)
                    or not completion[0]["content"].strip()
                    or not any(m.get("role") == "user" for m in prompt)):
                raise ValueError(f"{path.name} 第{line_no}行不是预期的prompt/completion对话格式")
            rows.append((prompt, completion[0]["content"].strip()))
    if not rows:
        raise ValueError(f"{path.name} 为空")
    return rows


def make_dataset(chat_rows, tokenizer, label):
    """Build textual prompt/completion with identical non-thinking prompt at inference."""
    items = []
    skipped = 0
    for prompt_msgs, answer in chat_rows:
        # The same call must be used during inference for alignment.
        prefix = tokenizer.apply_chat_template(
            prompt_msgs,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        # Append Qwen's end-of-message token to supervise a natural stop.
        completion = answer + "<|im_end|>"
        full_ids = tokenizer(prefix + completion, add_special_tokens=False)["input_ids"]
        prompt_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
        if len(full_ids) > MAX_LENGTH or len(prompt_ids) >= MAX_LENGTH - 4:
            skipped += 1
            continue
        items.append({"prompt": prefix, "completion": completion})
    if not items:
        raise ValueError(f"{label}：所有数据均超出{MAX_LENGTH} tokens，请先缩短上下文")
    print(f"{label}: 原始{len(chat_rows)}条，使用{len(items)}条，超长跳过{skipped}条")
    return Dataset.from_list(items)


def main():
    torch.manual_seed(SEED)
    if not torch.cuda.is_available():
        raise RuntimeError("未检测到CUDA，训练未启动")
    print("GPU:", torch.cuda.get_device_name(0))
    print("HF_HOME:", os.environ["HF_HOME"])
    print("训练将从原始Qwen3 1.7B开始，绝不继承V2 LoRA")

    raw_train = load_chat_rows(TRAIN_FILE)
    raw_valid = load_chat_rows(VALID_FILE)
    print(f"读取完毕：训练{len(raw_train)}条，验证{len(raw_valid)}条")

    model, tokenizer = FastModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_LENGTH,
        dtype=torch.float16,
        load_in_4bit=True,
    )
    model = FastModel.get_peft_model(
        model,
        r=8,
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        use_gradient_checkpointing="unsloth",
        random_state=SEED,
    )

    train_ds = make_dataset(raw_train, tokenizer, "训练集")
    valid_ds = make_dataset(raw_valid, tokenizer, "验证集")

    args = SFTConfig(
        output_dir=str(CHECKPOINT_DIR),
        max_length=MAX_LENGTH,
        num_train_epochs=2.0,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=1,
        learning_rate=2e-5,
        warmup_ratio=0.05,
        lr_scheduler_type="linear",
        fp16=True,
        bf16=False,
        optim="adamw_torch",
        gradient_checkpointing=True,
        max_grad_norm=1.0,
        logging_steps=5,
        eval_strategy="epoch",
        save_strategy="no",
        report_to="none",
        packing=False,
        completion_only_loss=True,
        assistant_only_loss=False,
        seed=SEED,
        data_seed=SEED,
    )
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        args=args,
        processing_class=tokenizer,
    )

    # Verify the final collator masks the user/system prompt, leaving only assistant.
    sample = trainer.data_collator([trainer.train_dataset[0]])
    labels = sample["labels"][0].tolist()
    token_ids = sample["input_ids"][0].tolist()
    supervised_ids = [tok for tok, target in zip(token_ids, labels) if target != -100]
    masked_count = sum(x == -100 for x in labels)
    if not supervised_ids or masked_count == 0:
        raise RuntimeError("发现训练标签异常：没有正确屏蔽prompt，已取消训练")
    preview = tokenizer.decode(supervised_ids, skip_special_tokens=False)
    print(f"标签检查：prompt已屏蔽{masked_count} tokens；assistant监督{len(supervised_ids)} tokens")
    print("助手目标预览:", repr(preview[:120]))
    if train_ds[0]["completion"].removesuffix("<|im_end|>")[:3] not in preview:
        print("提示：预览未完整显示目标回复，请先核对是否存在分词边界问题")

    print("开始V3 QLoRA训练（2 epochs，包含验证集评估）...")
    result = trainer.train()
    eval_metrics = trainer.evaluate()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))

    report = {
        "base_model": MODEL_NAME,
        "train_samples": len(train_ds),
        "valid_samples": len(valid_ds),
        "epochs": 2,
        "train_loss": result.training_loss,
        "eval_loss": eval_metrics.get("eval_loss"),
        "cuda_memory_allocated_peak_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
        "history": trainer.state.log_history,
    }
    report_path = OUTPUT_DIR / "training_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=========== V3训练完成 ===========")
    print("训练Loss:", result.training_loss)
    print("验证Loss:", eval_metrics.get("eval_loss"))
    print("PyTorch显存峰值:", report["cuda_memory_allocated_peak_gb"], "GB")
    print("LoRA保存位置:", OUTPUT_DIR)
    print("报告:", report_path)


if __name__ == "__main__":
    main()
