# QWbot V2

QWbot V2 是一个面向 QQ 群场景的聊天机器人项目，目标是让模型理解多人聊天上下文，并以简短、自然的群友口吻进行回复。

## 项目信息

| 项目 | 说明 |
| --- | --- |
| 目标 | QQ 群聊天机器人 |
| 模型 | Qwen4B（配置默认使用 `Qwen/Qwen3-4B`） |
| 训练方式 | LoRA / SFT |
| Alpha 训练集 | 1350 条 |
| Alpha 验证集 | 150 条 |

## 数据来源

- 真实 QQ 群聊天
- 人工增强场景
- gold 审核数据

## 数据原则

真实 QQ 群行为优先。

抗吧/meme 只作为辅助，不应覆盖真实群聊的表达习惯、回复时机和上下文逻辑。

## 仓库内容

- `configs/`：Qwen4B LoRA/SFT 推荐配置。
- `scripts/`：Alpha 数据集整理脚本，不包含旧实验脚本。
- `data/`：公开统计报告和 5 条虚构脱敏格式样例，不包含完整训练数据。
- `evaluation/`：公开评测用例格式示例。
- `docs/`：数据流水线和训练环境说明。

## 快速开始

创建环境并安装依赖：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

完整训练数据和元数据应保存在仓库之外。需要重新整理私有数据时，请显式指定输入与输出路径：

```bash
python scripts/prepare_qwbot_v2_alpha_dataset.py \
  --input /secure/path/qwbot_v2_expansion_sft_v1.jsonl \
  --metadata /secure/path/qwbot_v2_expansion_metadata.jsonl \
  --train-output /secure/output/qwbot_v2_alpha_train.jsonl \
  --val-output /secure/output/qwbot_v2_alpha_val.jsonl \
  --report-output /secure/output/qwbot_v2_alpha_report.json
```

该脚本只做数据校验、去重、分组切分和导出，不会启动训练。更多信息见 [数据流水线](docs/DATA_PIPELINE.md) 与 [训练说明](docs/TRAINING.md)。

## 隐私说明

本仓库不发布真实聊天原文、完整训练集、群标识或用户标识。`data/data_example.jsonl` 中的 5 条记录为虚构脱敏示例，仅用于展示数据格式。公开前仍应检查 Git 历史，避免曾经提交过私有数据、模型权重或访问凭据。
