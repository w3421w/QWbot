# 数据整理脚本

本目录只保留 QWbot V2 Alpha 的数据整理脚本：

- `prepare_qwbot_v2_alpha_dataset.py`

脚本读取 SFT JSONL 及其元数据 sidecar，执行结构检查、保守质量过滤、重复检查和分组切分，随后输出训练集、验证集与统计报告。它不会下载模型，也不会启动训练。

## 输入格式

SFT JSONL 每行应为一个对象，并且 `messages` 严格包含 `system`、`user`、`assistant` 三条消息。元数据 JSONL 与输入按记录顺序对应，用于读取 `source_id`、`category` 和 `source`，以降低同源上下文跨集合泄漏的风险。

## 使用方式

复制到本仓库后，脚本内置默认路径会相对于 `scripts/` 解析。为避免把真实数据写入公开仓库，建议始终显式传入全部路径：

```bash
python scripts/prepare_qwbot_v2_alpha_dataset.py \
  --input /secure/path/input.jsonl \
  --metadata /secure/path/metadata.jsonl \
  --train-output /secure/output/train.jsonl \
  --val-output /secure/output/validation.jsonl \
  --report-output /secure/output/report.json \
  --val-ratio 0.10 \
  --seed 20261013
```

不要将输入文件、元数据、生成的完整训练集或验证集提交到公开仓库。

注意：脚本生成的完整报告包含用于审计的逐条预览字段，也应按私有产物处理。`data/qwbot_v2_alpha_report.json` 是为公开发布单独清理过的副本，不要用新生成的私有报告直接覆盖它。
