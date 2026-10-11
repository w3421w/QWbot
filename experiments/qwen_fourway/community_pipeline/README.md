# QWbot 社区语料离线工程

仅处理用户合法提供的离线 JSON/JSONL 导出。规则分类是候选标签，所有记录 `trainable=false`。未导入真实 B 站或小红书样本；目前只完成代码与模拟测试。不得把评论、弹幕和直播聊天拼成自动 SFT。

## 输入格式

每个文件为 JSON 数组、`{"records":[...]}` 或逐行 JSONL。B 站评论必填 `id,text,video_id`（或真实 `thread_id`）；小红书必填 `id,text,post_id`。可选 `parent_id,root_id,author_id,order,created_at,content_type,source_license,source_url`。`parent_id` 必须是导出中的真实父 ID；缺失时留空。`created_at` 原样保留，不推断。默认 `content_type=comment`；B 站另支持 `danmaku` 和 `livestream_chat`，小红书支持 `post`。普通用户 ID 会确定性散列；审核 CSV 不输出昵称和 URL。输入文件中不要放平台凭证。

## WSL 终端试点命令

```bash
cd /home/alex/Qwen_Finetune
P=experiments/qwen_fourway/community_pipeline
.venv/bin/python "$P/02_parse_bilibili.py" --input /path/to/authorized_bilibili_export.jsonl --output "$P/output_community_pilot/bilibili_parsed.jsonl" --stats "$P/output_community_pilot/bilibili_parse_stats.json"
.venv/bin/python "$P/04_classify_topics.py" --input "$P/output_community_pilot/bilibili_parsed.jsonl" --output "$P/output_community_pilot/bilibili_classified.jsonl"
.venv/bin/python "$P/06_deduplicate.py" --input "$P/output_community_pilot/bilibili_classified.jsonl" --blind-test datasets/QWbot_V44_真实群聊影子测试_匿名31题.jsonl --output "$P/output_community_pilot/bilibili_duplicates.json"
.venv/bin/python "$P/03_validate_community.py" --input "$P/output_community_pilot/bilibili_classified.jsonl" --parse-stats "$P/output_community_pilot/bilibili_parse_stats.json" --duplicate-report "$P/output_community_pilot/bilibili_duplicates.json" --output "$P/output_community_pilot/bilibili_quality.json"
.venv/bin/python "$P/07_export_review.py" --input "$P/output_community_pilot/bilibili_classified.jsonl" --output "$P/output_community_pilot/bilibili_review.csv"
.venv/bin/python "$P/05_parse_xiaohongshu.py" --input /path/to/authorized_xiaohongshu_export.jsonl --output "$P/output_community_pilot/xiaohongshu_parsed.jsonl" --stats "$P/output_community_pilot/xiaohongshu_parse_stats.json"
```

Outputs use exclusive creation and fail if already present. Use a fresh output name for each run. Real pilot order: B 站 LOL/抗吧、CS2、VALORANT、炉石/酒馆战棋、NBA、足球、虚拟主播、普通热梗，随后小红书日常讨论。Start with a small authorized batch, verify source rights, structure, privacy and quality, then consider larger import. Split by `source_group_id`, which uses actual video/post/thread ID; a technical split never grants training rights.

## Hugging Face 公开数据小样本（08_fetch_public_pilot.py）

`08_fetch_public_pilot.py` 只允许 `Midsummra/bilibilicomment` 和
`helloollel/chat_dataset_bilibili`。默认只访问 Hugging Face 元数据；不读取
完整仓库，不使用账户 Token。`--download-pilot` 才访问
`datasets-server` 的 `first-rows` 预览接口，每个来源最多 100 行，响应上限
2 MiB、请求有超时。输出文件采用独占创建，已存在时失败。记录中的
`source_license` 是数据集声明；`license_status` 保持待核验，不能据此放行
底层评论或训练。

适配约束：Midsummra 记录是独立 B 站评论候选；helloollel 保留
`instruction`/`input`/`output`，并标记 `relation_type=derived_instruction_response`。
两者都不推断 `thread_id`、`parent_comment_id` 或 `speaker_id`，且所有记录
`trainable=false`。分类是规则候选，`classification_evidence` 保存命中的规则；
未命中记录保留 `topics=["unknown"]`，人工复核可用 `--manual-overrides`。

### WSL 真实命令（本次不执行）

```bash
cd /home/alex/Qwen_Finetune
P=experiments/qwen_fourway/community_pipeline
META_OUT="$P/output_public_metadata_$(date +%Y%m%d_%H%M%S)"
OUT="$P/output_public_pilot_$(date +%Y%m%d_%H%M%S)"

# 1) 仅检查两个来源的元数据（不会获取样本）
.venv/bin/python "$P/08_fetch_public_pilot.py" --source all --output-dir "$META_OUT"

# 2) 使用全新目录获取每个来源最多100条头部预览（非随机样本）
#    --download-pilot本身会再次保存元数据，因此不能复用META_OUT。
.venv/bin/python "$P/08_fetch_public_pilot.py" --source all --download-pilot --limit 100 --output-dir "$OUT"

# 3) 规则解析/分类、去重、质量统计、人工审核抽样
for S in Midsummra__bilibilicomment helloollel__chat_dataset_bilibili; do
  .venv/bin/python "$P/04_classify_topics.py" \
    --input "$OUT/${S}.pilot.jsonl" --output "$OUT/${S}.classified.jsonl"
  .venv/bin/python "$P/06_deduplicate.py" \
    --input "$OUT/${S}.classified.jsonl" --output "$OUT/${S}.duplicates.json"
  .venv/bin/python "$P/03_validate_community.py" \
    --input "$OUT/${S}.classified.jsonl" \
    --duplicate-report "$OUT/${S}.duplicates.json" \
    --output "$OUT/${S}.quality.json"
  .venv/bin/python "$P/07_export_review.py" \
    --input "$OUT/${S}.classified.jsonl" --output "$OUT/${S}.review.csv"
done
```

`08_fetch_public_pilot.py` 输出的 pilot JSONL 已完成公开数据适配与隐私扫描，
因此直接进入既有分类、去重、质量统计和审核导出步骤；不会生成任何 SFT
训练文件，也不会修改旧数据或 31 题盲测。


注意：first-rows为数据集前100条预览，非随机抽样；config/split通过/splits查询，不预设default/train。
