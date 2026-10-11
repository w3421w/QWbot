# 抗吧原始语料清洗工程 V1

本目录是只读原始数据的第一阶段审计管线。原始文件位于 `datasets/kangba_raw/data/`，脚本不会修改它们，也不会写入既有 `datasets/kangba_*` SFT 文件。

## 已确认的原始结构

105 个 JSON 文件均为列表，共 5136 个帖子。每个帖子只有：`标题`、`链接`、`楼主内容`、`回复列表`；`回复列表` 的 118446 个元素均为字符串。`链接` 可解析 `/p/<数字>` 帖子 ID。原始数据没有回复 ID、作者、时间、楼层、引用或回复对象字段，因此所有回复的 `reply_relation` 必须是 `unknown`，不能推断为回复楼主或楼中楼。

## 运行

```bash
python 01_inspect_raw.py --check-only --sample-posts 100 --seed 42 --output-dir output
python 02_clean_kangba.py --check-only
python 02_clean_kangba.py --sample-posts 100 --seed 42 --output-dir output
python 03_build_sft_candidates.py --output-dir output
python 04_export_review_samples.py --output-dir output
```

`02` 对空回复、明确广告、明确严重威胁拒绝；联系方式脱敏；完全重复保留但加 `exact_duplicate`，近似重复加 `near_duplicate_risk`。文本自然度和回复关系分开评分；关系因缺字段为 0，非拒绝样本默认 `pending_review`，许可默认 `unconfirmed`。真实回复统一标为 `provenance_type=real_reply`，没有模型补写。

`03` 读取 31 题独立测试集的上下文、问题和示例答案，建立精确/近重复隔离清单，不导出正式训练集。`04` 只导出人工审核 CSV。

输出包括结构报告、清洗前后统计、候选/拒绝 JSONL、测试重叠审核清单和 `review_samples.csv`。
