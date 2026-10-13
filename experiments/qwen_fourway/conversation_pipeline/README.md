# QWbot 第二阶段：多人对话语料扩充

本目录是独立的多人对话数据工程。目标模型为 Qwen3-4B-Instruct-2507；本阶段不修改抗吧 V2.1、不训练模型。SmoothConv（`qualialabsAI/SmoothConv`）声明为 **CC BY-NC 4.0**，候选数据仅供非商业研究用途，`trainable` 固定为 `false`。

## 本地运行（正常 WSL 终端）

本次 Codex 沙箱不联网。用户确认网络和许可后，在项目根目录执行：

```bash
python experiments/qwen_fourway/conversation_pipeline/01_fetch_smoothconv_json.py --scenario social --list-only
python experiments/qwen_fourway/conversation_pipeline/01_fetch_smoothconv_json.py --scenario social --download --max-files 30 --max-total-mb 20 --output-dir datasets/conversation_raw/smoothconv_social_pilot
python experiments/qwen_fourway/conversation_pipeline/02_parse_smoothconv.py --input-dir datasets/conversation_raw/smoothconv_social_pilot --output-dir experiments/qwen_fourway/conversation_pipeline/output_smoothconv_social_pilot
python experiments/qwen_fourway/conversation_pipeline/03_validate_conversations.py --input-dir experiments/qwen_fourway/conversation_pipeline/output_smoothconv_social_pilot
```

下载器只查询官方清单和下载 `.json` 标注，不下载 `.wav`、模型或整个仓库；默认必须显式 `--download` 才写盘。原始文件存于 `datasets/conversation_raw/smoothconv_json/`，带 SHA256 与来源记录，不覆盖已有不同内容。

## 结构约定

解析器实际检查 `instances`、`channelIndex`、`attributes.speaker`、`start`、`end`、`text` 字段。发言人只在单段对话内匿名化；按时间保留重叠；无证据不建立回复关系；`speech_transcript` 与网络群聊来源分开标记。空白、转写垃圾和明显隐私风险会过滤或标记。输出为 `output_smoothconv_pilot/` 下的 `conversations.jsonl`、`messages.jsonl`、`quality_statistics.json`、脱敏 `review_samples.csv`（UTF-8-SIG）和 `pilot_report.md`。

## 后续适配器接口

获得明确授权后，可为 QQ 群导出、Discord、Telegram、Matrix/IRC 公开授权记录、带父评论 ID 的 B 站楼中楼，以及 NBA、足球、CS2、无畏契约、DOTA2 讨论串分别实现输入适配器。适配器只需映射到统一消息字段，并保留 `source`、`source_file`、`source_license`、`license_status`、`provenance_type` 和回复证据。公开可阅读不等于允许训练；禁止擅自抓取私人群聊。父评论 ID 存在且可验证时才能填写 `reply_to_id`。

## 测试

```bash
python -m unittest discover -s experiments/qwen_fourway/conversation_pipeline/tests -v
python -m py_compile experiments/qwen_fourway/conversation_pipeline/*.py
```

31 题测试集保持独立，不复制到本目录；解析后的对话按 `conversation_id` 整段划分，禁止跨训练/验证。

## MPDD（中文电视剧台词）

MPDD 使用官方 GitHub 仓库 `ntunlplab/Dialogue-MPDD` 的 `mpdd.zip`。Codex 阶段不联网、不下载；默认抓取命令只做来源和文件大小检查，只有追加 `--download` 才会下载。下载器限制总大小、拒绝覆盖已有 ZIP、记录 SHA256 和来源，并在写盘前检查 ZIP 路径穿越、异常压缩比和解压总大小。

解析输出固定标记为 `provenance_type=scripted_tv_dialogue`，每段对话内匿名化 speaker，保留原始顺序、listener 语义和空台词；listener 不会被推断为 `reply_to_id`，没有消息级证据时 `reply_relation=unknown`。不生成 SFT 问答，不补时间戳，所有 conversation/message 的 `trainable` 都是 `false`。训练及再分发权利保持待确认。

### 正常 WSL 终端运行

```bash
# 1) 默认只检查官方来源和文件大小（不下载）
python experiments/qwen_fourway/conversation_pipeline/04_fetch_mpdd.py \
  --output-dir datasets/conversation_raw/mpdd --max-total-mb 512

# 2) 权利确认后才执行真实下载；不会覆盖已存在 mpdd.zip
python experiments/qwen_fourway/conversation_pipeline/04_fetch_mpdd.py \
  --download --output-dir datasets/conversation_raw/mpdd --max-total-mb 512

# 3) 从本地 mpdd.zip 读取 dialogue.json / metadata.json
python experiments/qwen_fourway/conversation_pipeline/05_parse_mpdd.py \
  --input-dir datasets/conversation_raw/mpdd \
  --output-dir experiments/qwen_fourway/conversation_pipeline/output_mpdd_pilot

# 4) 真实数据审计（按原始对话及可用上级来源 ID 分组切分）
python experiments/qwen_fourway/conversation_pipeline/06_validate_mpdd.py \
  --output-dir experiments/qwen_fourway/conversation_pipeline/output_mpdd_pilot \
  --test-set datasets/QWbot_V44_真实群聊影子测试_匿名31题.jsonl
```

审计会生成 `conversations.jsonl`、`messages.jsonl`、`quality_statistics.json`、UTF-8-SIG 的 `real_conversation_review.csv`、`license_review.md` 和 `pilot_report.md`。若输出目录中已有同名审计文件，脚本拒绝覆盖。

## IRC Disentanglement（真实网络群聊，试点）

IRC 适配器只读取本地已选的配套 `.raw.txt`、`.ascii.txt`、`.annotation.txt` 文件，不在 Codex 阶段联网。由于当前工作区没有官方仓库副本，字段解释以下载后官方 README/数据说明为准；解析器采取保守规则：消息行只接受制表符分隔的 `message_id, timestamp, speaker, text`（也兼容缺 timestamp 的三列）；不填充缺失时间戳。标注只有在行内明确写出 `reply<TAB>source_id<TAB>target_id`（或同等 `source target reply`）时才建立关系。无明确方向的 ID 对会记录为 `unresolved_edge_direction_or_semantics`，绝不按相邻消息推断。自连接保留为 `reply_relation=self_start`；一条消息多个目标保留在 `reply_to_ids`；找不到上下文 ID 记录 `out_of_context_reference`。系统发言保留并标记 `system_message=true`。原始昵称只用于内存中的匿名化映射，不写入输出审核 CSV。

### 正常 WSL 终端的小规模运行

先根据官方仓库文档选择少量同一日志的文件名；下面命令只检查白名单，不联网下载：

```bash
python experiments/qwen_fourway/conversation_pipeline/07_fetch_irc.py \
  --file data/<small-log>.raw.txt \
  --file data/<small-log>.ascii.txt \
  --file data/<small-log>.annotation.txt
```

确认来源、许可和大小后，执行真实下载（最多 6 个文件、50 MiB，已有文件拒绝覆盖）：

```bash
python experiments/qwen_fourway/conversation_pipeline/07_fetch_irc.py --download \
  --file data/<small-log>.raw.txt \
  --file data/<small-log>.ascii.txt \
  --file data/<small-log>.annotation.txt \
  --output-dir datasets/conversation_raw/irc_disentanglement
python experiments/qwen_fourway/conversation_pipeline/08_parse_irc.py
python experiments/qwen_fourway/conversation_pipeline/09_validate_irc.py \
  --test-set datasets/QWbot_V44_真实群聊影子测试_匿名31题.jsonl
```

输出包括 `conversations.jsonl`、`messages.jsonl`、`parse_errors.json`、`quality_statistics.json`、脱敏 `real_conversation_review.csv`、`license_review.md` 和 `pilot_report.md`。训练/验证应按原始日志或上级来源分组；IRC 输出固定 `trainable=false`，不生成 SFT 问答。
