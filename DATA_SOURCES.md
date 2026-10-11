# QWbot V6 第一批真实数据来源计划

当前扫描已发现本地抗吧/贴吧社区数据、B 站评论示例和多份中文对话/评测 JSONL；未发现文件名或元数据能确认其为 LCCC。所有数据进入 pipeline 前先运行 `30_v6_real_dataset_probe.py`，再人工确认授权、隐私和字段。

| 名称 | 来源 | 格式 | 预计转换方式 |
|---|---|---|---|
| 中文聊天数据（优先） | LCCC（公开许可版本）或自有、已匿名化聊天记录 | JSON/JSONL | 探针识别 `messages`/`conversation` 或 prompt-response；映射为 dialogue schema，去重并做隐私清理 |
| B站评论/回复数据（优先） | 经授权导出的 B 站评论、楼中楼回复 | JSON/JSONL/CSV | 识别评论文本、用户、回复关系和时间；转为 comment/reply 样本，匿名化用户字段 |
| 游戏社区数据（优先） | 经授权的游戏论坛/社区帖子与评论 | JSON/JSONL/CSV | 识别帖子、评论、作者和上下文；按社区语境转为抗压风格训练候选，保留原始证据链 |

## 现有本地数据

`dataset_inventory.json` 是本次扫描结果。现有 `datasets/` 中的抗吧/贴吧文件可作为社区评论候选，`community_pipeline/external_source_index/bilibili_comments_crawl/` 含 B 站评论示例；现有中文聊天 JSONL 可先做 schema 探针。是否可直接训练仍需通过质量和隐私检查，不能仅凭文件名认定为 LCCC。
