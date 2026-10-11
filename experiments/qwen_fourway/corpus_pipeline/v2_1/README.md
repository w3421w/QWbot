# 抗吧语料清洗 V2.1 结构修复版

只使用 Python 标准库；不联网、不训练、不导出 `messages`。V2.1 不修改 V2 或任何旧数据。

`source_group_id` 固定为单帖来源组，文本重复使用独立的 `text_duplicate_group`，近似重复仅作为高信息量泄漏约束。重复连接组件只用于 train/validation 隔离，不改变来源组，也不截断组件。短梗保留并记录频次。

原始标题、正文、回复和 URL 只写入权限为 0600 的 `internal_audit_restricted.jsonl`；下游候选和审查 CSV 不含这些字段，仅保留脱敏文本、哈希和来源索引。正则无法识别全部个人信息，仍需人工审核。

```bash
python 01_clean_all.py --check-only
python 01_clean_all.py --sample-posts 100 --seed 20261009 --output-dir output_sample_100 --review-dir output_review_sample_100
python 03_export_review.py --input-dir output_sample_100 --output-dir output_review_sample_100
python 04_validate_outputs.py --input-dir output_sample_100 --review-dir output_review_sample_100
python -m unittest discover -s tests -v
```

所有关系为 `unknown`，许可证为 `unconfirmed_conflicting_source_metadata`，`trainable=false`，正式可训练数量为 0。31 题测试集命中及其重复连接组件进入 `test_quarantine`。
