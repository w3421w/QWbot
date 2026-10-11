import os
import json
import random
import torch

os.environ.setdefault(
    "HF_HOME",
    os.path.expanduser("~/Qwen_Finetune/hf_cache")
)

from datasets import Dataset
from unsloth import FastModel
from trl import SFTTrainer, SFTConfig

MODEL = "unsloth/Qwen3-1.7B-unsloth-bnb-4bit"
OUTPUT = "outputs/kangba_v2_lora"

# 从完整训练集中随机抽取2000条
records = []
with open("datasets/kangba_train.jsonl", encoding="utf-8") as f:
    for line in f:
        records.append(json.loads(line))

print("完整训练集:", len(records))

random.seed(3407)
samples = random.sample(records, min(2000, len(records)))
dataset = Dataset.from_list(samples)

print("本次实验候选:", len(dataset))

model, tokenizer = FastModel.from_pretrained(
    model_name=MODEL,
    max_seq_length=512,
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
    random_state=3407,
)

args = SFTConfig(
    output_dir="outputs/kangba_v2_checkpoints",
    max_length=512,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=2,
    max_steps=100,
    learning_rate=5e-5,
    warmup_steps=5,
    fp16=True,
    bf16=False,
    optim="adamw_torch",
    logging_steps=10,
    save_strategy="no",
    report_to="none",
    packing=False,
    completion_only_loss=True,
    assistant_only_loss=False,
    gradient_checkpointing=True,
    seed=3407,
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=args,
    processing_class=tokenizer,
)

print("开始第一次正式语料微调...")
result = trainer.train()

model.save_pretrained(OUTPUT)
tokenizer.save_pretrained(OUTPUT)

print("训练完成")
print("训练Loss:", result.training_loss)
print("LoRA保存位置:", OUTPUT)
print(
    "PyTorch显存峰值:",
    round(torch.cuda.max_memory_allocated() / 1024**3, 2),
    "GB"
)
