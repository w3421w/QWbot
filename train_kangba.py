import os
os.environ.setdefault("HF_HOME", os.path.expanduser("~/Qwen_Finetune/hf_cache"))

import json
import torch
from pathlib import Path
from datasets import Dataset
from unsloth import FastModel
from trl import SFTTrainer, SFTConfig

MODEL_NAME = "unsloth/Qwen3-1.7B-unsloth-bnb-4bit"
DATA_PATH = Path("datasets/kangba_smoke_test.jsonl")
OUTPUT_DIR = "outputs/kangba_smoke_lora"

print("========== 开始抗吧Qwen微调 ==========")

# 1. 加载已经下载的4-bit模型
model, tokenizer = FastModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=384,
    dtype=torch.float16,
    load_in_4bit=True,
)

# 2. 添加LoRA适配器
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
    random_state=3407,
)

# 3. 读取训练数据
# 转成对话式prompt-completion格式，
# 只对assistant回答计算训练损失。
records = []

with DATA_PATH.open("r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        item = json.loads(line)
        messages = item["messages"]
        assert messages[-1]["role"] == "assistant"

        records.append({
            "prompt": messages[:-1],
            "completion": [messages[-1]],
        })

dataset = Dataset.from_list(records)
print(f"训练样本数量: {len(dataset)}")

# 4. 训练参数
args = SFTConfig(
    output_dir="outputs/kangba_checkpoints",
    max_length=384,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=2,
    max_steps=3,
    learning_rate=1e-4,
    warmup_steps=0,
    fp16=True,
    bf16=False,
    optim="adamw_torch",
    logging_steps=1,
    save_strategy="no",
    report_to="none",
    packing=False,
    completion_only_loss=True,
    assistant_only_loss=False,
    gradient_checkpointing=True,
    seed=3407,
)

# 5. 创建训练器
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=args,
    processing_class=tokenizer,
)

# 6. 检查是否存在有效训练标签
first = trainer.train_dataset[0]
labels = first.get("labels")
if labels is not None:
    assert any(x != -100 for x in labels), \
        "训练标签全部被屏蔽，请检查数据格式"

print("开始执行QLoRA训练...")
result = trainer.train()

# 7. 保存LoRA适配器
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("\n========== 训练完成 ==========")
print("最终训练损失:", result.training_loss)
print("LoRA保存位置:", OUTPUT_DIR)
print(
    "GPU显存峰值:",
    round(torch.cuda.max_memory_allocated() / 1024**3, 2),
    "GB"
)
