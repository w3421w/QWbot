# 抗吧语料全量清洗 V2

运行环境：Python 标准库，无网络、模型或 GPU。V2 仅读取 105 个原始 JSON 和冻结的 31 题测试集，不改动 V1/V1.1、既有训练数据或权重。输出目录一旦存在，清洗命令直接报错；人工审核 CSV 只写入预先由清洗命令创建的空目录。

```bash
python 01_clean_all.py --check-only
python 01_clean_all.py --sample-posts 100 --seed 20261009 --output-dir output_sample_100 --review-dir output_review_sample_100
python 03_export_datasets.py --input-dir output_sample_100 --output-dir output_review_sample_100
python 04_validate_outputs.py --input-dir output_sample_100 --review-dir output_review_sample_100
python 01_clean_all.py
python 03_export_datasets.py
python 04_validate_outputs.py
```

## 规则与边界

- 每个回复保存原始文本、NFKC 与空白规范化文本、两个文本哈希、帖子 ID、源文件、帖内序号、标题、正文和原始上下文。隐私风险行隔离至权限为 `0600` 的 `rejected.jsonl`；人工 CSV 只展示脱敏文本。输出目录权限为 `0700`。
- 空白、明确广告、严重威胁、机器重复、乱码、过长刷屏、隐私风险与完全重复的后续出现为 `REJECT`，保留全部原因及来源。近似重复采用固定的 64 位字符三元组 SimHash、四个 16 位桶、汉明距离最多 3 位，并以 `SequenceMatcher` 比例至少 0.90 复核；短梗小于 8 字符只做精确去重。该方法可复现，但不能发现所有语义改写。
- A/B/C 由 `02_quality_grade.py` 的文本、风格、上下文与惩罚分组成。短而有价值的梗不会单凭字数拒绝；缺上下文或近似重复进入 C 级复核。未经证实的个人传闻标记为 C，不把传闻写为事实。
- SFT 待审候选只来自 A/B 风格文本，且具有标题/正文字符重合与问题/回答线索。该启发式只表示可能相关，不表示回复对象已验证。所有 `reply_relation` 仍为 `unknown`。原始数据无作者、时间、楼层和引用；不重建多轮对话。
- 对测试集的消息、上下文和人工示例答复做规范化精确与字符三元组召回后的近重复比较。命中回复、帖子及其去重关联的来源组整体隔离。训练/验证按重复关联的帖子组哈希切分，不跨组。`test_overlap_review.jsonl` 记录命中和整组隔离证据。
- 原始 README 的前置许可证写 MIT，正文引用写 CC BY-NC 4.0，状态统一为 `unconfirmed_conflicting_source_metadata`。自动质量等级、人工审核状态、关系状态和许可状态四个字段互不替代。正式可训练数量恒为 0，程序不导出 `messages`。

## 人工审核与后续 SFT

`output_review/stratified_human_review.csv` 用 UTF-8-SIG 编码，每等级固定种子抽样，人工判断字段留空。优先审查隐私/传闻误判、C 级上下文不足、A/B 级风格价值和高优先级 SFT 对应关系。若未来核实来源许可、完成逐条人工审核并取得可验证的输入输出对应证据，再独立实现训练转换器；该转换器须只读取获授权且人工通过的来源组，维持原帖子组切分和测试隔离，按 Qwen3-4B-Instruct-2507 的 `messages` 格式导出，并在训练前再次检查许可、关系、去重与测试集。当前 V2 不产生可训练文件。
