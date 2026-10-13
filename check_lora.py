import torch
from unsloth import FastModel
from peft import PeftModel

BASE = "unsloth/Qwen3-1.7B-unsloth-bnb-4bit"
ADAPTER = "outputs/kangba_smoke_lora"

model, tokenizer = FastModel.from_pretrained(
    model_name=BASE,
    max_seq_length=384,
    dtype=torch.float16,
    load_in_4bit=True,
)

model = PeftModel.from_pretrained(model, ADAPTER)
model.eval()

messages = [
    {"role": "system", "content": "你是QQ群里的抗吧串子，回复简短。"},
    {"role": "user", "content": "我上单0/8，这把全怪打野"}
]

inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    enable_thinking=False,
    return_tensors="pt",
).to("cuda")

with torch.inference_mode():

    # 禁用LoRA
    with model.disable_adapter():
        original = model(input_ids=inputs).logits[:, -1, :].float()

    # 启用LoRA
    finetuned = model(input_ids=inputs).logits[:, -1, :].float()

difference = (finetuned - original).abs()

print("\n========== LoRA检查结果 ==========")
print("最大Logits差异:", difference.max().item())
print("平均Logits差异:", difference.mean().item())

if difference.max().item() > 0:
    print("结果：LoRA已经改变模型输出数值")
else:
    print("结果：没有检测到数值差异，需要进一步排查")
