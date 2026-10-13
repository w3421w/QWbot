# QWbot V2 训练说明

本页记录推荐环境与超参数，不附带或触发训练任务。完整数据应放在公开仓库之外。

## 云服务器环境

- Python：建议 3.10 或 3.11。
- CUDA：版本需与服务器 NVIDIA 驱动及 PyTorch 构建匹配。
- PyTorch：先安装与 CUDA 匹配的 GPU 版本，并确认 `torch.cuda.is_available()` 为 `True`。
- Transformers：负责加载 Qwen4B 模型与 tokenizer。
- PEFT：负责注入和保存 LoRA adapter。
- TRL：可用于 SFT 训练流程。
- Accelerate：用于设备配置、混合精度和多卡启动。
- bitsandbytes：显存受限时可选；使用前需确认平台和 CUDA 兼容性。

基础环境：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

生产环境中建议根据云服务器 CUDA 版本先安装对应的 PyTorch wheel，再安装其余依赖。仓库的 `requirements.txt` 不锁定版本，以避免默认绑定到错误的 CUDA 构建；正式复现实验时应额外保存实际的包版本和 GPU 信息。

## Qwen4B LoRA 推荐配置

| 参数 | 推荐值 |
| --- | --- |
| LoRA rank (`r`) | 16 |
| LoRA alpha | 32 |
| Learning rate | `1.5e-5` |
| Epoch | 3 |
| Max sequence length | 1024 |

完整参考配置位于 `configs/qwen4b_lora_config.yaml`。其中默认基础模型标识为 `Qwen/Qwen3-4B`，训练方式为 LoRA + SFT；如使用其他 Qwen 4B 变体，应同步修改模型路径、chat template 和目标模块，并在实验记录中注明。

## 数据准备

训练和验证文件均使用 JSONL，每行包含 `messages` 数组。训练前应确认：

1. 完整数据位于仓库之外，路径已写入本地配置副本。
2. tokenizer 的 chat template 能正确处理 `system`、`user`、`assistant`。
3. 只对 assistant 回复计算目标损失，或明确记录所用的 loss mask 策略。
4. 序列截断不会系统性删除最后的 assistant 回复。
5. 训练集为 1350 条，验证集为 150 条，且同源上下文没有跨集合泄漏。

## 显存与精度

配置默认使用 BF16、单卡 batch size 1、梯度累积 16 和 gradient checkpointing。若 GPU 不支持 BF16，可在验证数值稳定性后改用 FP16。显存不足时可以评估量化加载，但量化 LoRA 属于不同实验设置，应单独记录，不要与标准 LoRA 结果混为一谈。

## 训练后检查

- 保存 adapter、tokenizer、配置文件和依赖版本，不要提交模型权重到 Git。
- 同时记录随机种子、有效 batch size、学习率曲线和验证损失。
- 使用仓库外的隔离测试集检查回复时机、群聊自然度、事实边界和隐私复现风险。
- 在公开任何生成样例前再次进行人工隐私审查。
