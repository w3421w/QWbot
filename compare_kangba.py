import torch
from pathlib import Path
from unsloth import FastModel
from peft import PeftModel

BASE = "unsloth/Qwen3-1.7B-unsloth-bnb-4bit"
ADAPTER = "outputs/kangba_smoke_lora"

assert Path(ADAPTER, "adapter_config.json").exists(), "找不到LoRA配置文件"

model, tokenizer = FastModel.from_pretrained(
    model_name=BASE,
    max_seq_length=384,
    dtype=torch.float16,
    load_in_4bit=True,
)

messages = [
    {
        "role": "system",
        "content": "你是QQ群里的抗吧串子，回复简短，自然接梗。"
    },
    {
        "role": "user",
        "content": "我上单0/8，这把全怪打野"
    },
]

inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    enable_thinking=False,
    return_tensors="pt",
).to("cuda")

def reply(m):
    m.eval()
    torch.manual_seed(3407)
    with torch.inference_mode():
        result = m.generate(
            input_ids=inputs,
            max_new_tokens=80,
            do_sample=True,
            temperature=0.7,
            top_p=0.8,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(
        result[0][inputs.shape[-1]:],
        skip_special_tokens=True
    ).strip()

print("\n【原始Qwen回复】")
print(reply(model))

model = PeftModel.from_pretrained(model, ADAPTER)

print("\n【微调Qwen回复】")
print(reply(model))
