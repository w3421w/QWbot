# QWbot 四模型实验：Codex 项目交接与执行任务

> **用途**：将本文件交给 Codex，让其在用户现有 WSL2 项目中接管后续代码检查、模型推理问题修复和四模型对照评测。
>
> **更新时间**：2026-10-09。以下“已验证”仅指用户在本次对话中贴出的运行日志；Codex 必须检查本机文件和运行状态，不得把历史描述当作当前代码一定如此。
>
> **最高原则**：先检查、再最小修改、再运行测试。**不要擅自重新训练、删除缓存、修改旧 V4.4 适配器、接入真实 QQ 群发消息或重构整个项目。**

## 0. 给 Codex 的直接任务

你是一名负责本地大模型推理、评测与工程隔离的 Python 工程师。请在现有项目中完成以下工作：

1. **审计**四模型实验代码、模型路径、测试集、GPU 和当前补丁安装情况；先给出发现的问题与具体执行顺序。
2. **修复** `04_4b_thinking` 在思考未结束时将推理文字当成最终回复的情况；确认 Qwen3-4B-Thinking-2507 的真实聊天模板和特殊标记，正确解析思考段与最终答复；同时识别被 `max_new_tokens` 截断的运行。
3. **保持四组实验严格隔离**：01/02 使用相同 1.7B 基础模型但分别开启/关闭思考；03 是 4B Instruct 非思考；04 是 4B Thinking。每个实验独立输出，不覆盖其他组。
4. **构建/完善四模型对照评测**：复用现有 31 道匿名真实群聊题，对四组模型施加一致的聊天历史与当前发言，记录输出、时延、Token、显存、思考完成状态等；只在离线终端运行，不向 QQ 发送。
5. **提供可重复执行命令、验证脚本、结果文件与报告**，明确哪些测试实际在本机 GPU 上跑通、哪些仅通过静态检查。

完成每个阶段后报告结果，不要把代码“能编译”误写成“模型已通过 GPU 测试”。

## 1. 机器、环境、项目位置

| 项目 | 当前情况 |
|---|---|
| 主机 | Windows 11；WSL2，发行版名称 `QWbot` |
| Linux 用户 | `alex` |
| Linux 根项目 | `/home/alex/Qwen_Finetune`（`~/Qwen_Finetune`） |
| Windows 文件访问 | `\\wsl$\QWbot\home\alex\Qwen_Finetune` |
| Python 环境 | `/home/alex/Qwen_Finetune/.venv`，Python 3.10 |
| GPU | NVIDIA GeForce RTX 2070，8 GB 显存 |
| 主要生态 | PyTorch、Transformers、bitsandbytes、Hugging Face、Unsloth；旧 V4.4 使用 PEFT/QLoRA |
| 网络模型源 | 可使用 `HF_ENDPOINT=https://hf-mirror.com` |
| Hugging Face 缓存 | `HF_HOME="$HOME/Qwen_Finetune/hf_cache"` |

启动命令：

```bash
cd ~/Qwen_Finetune
source .venv/bin/activate
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME="$HOME/Qwen_Finetune/hf_cache"
```

Codex 应当首先执行**只读检查**：

```bash
pwd
nvidia-smi
python -c "import sys,torch,transformers;print(sys.version);print('torch=',torch.__version__);print('transformers=',transformers.__version__);print('cuda=',torch.cuda.is_available())"
find experiments/qwen_fourway -maxdepth 4 -type f | sort
```

如果当前 Codex 运行在 Windows PowerShell，而不是 WSL，先明确进入正确的 `QWbot` 发行版，**不要**在 Windows 同名目录里另外创建一套不连通的实验环境。

## 2. 用户目标、机器人定位与现有成果

### 最终目标

开发一个适用于中文 QQ 群的 AI 群友，口吻偏“抗压背锅吧”（抗吧）：自然、简短、有梗，善于处理反串、回旋镖、双标、前后矛盾和多人群聊关系；不限于 LOL。遇到真实求助、医学、法律问题应转换为谨慎、正常的回复，不能无依据胡说。

### 现有旧项目：不可破坏

- 旧基础模型：`unsloth/Qwen3-1.7B-unsloth-bnb-4bit`。
- V4.4 LoRA：`outputs/kangba_v44_lora`（**不得覆盖**）。
- V4.4 训练脚本可能位于项目根：`train_kangba_v44.py`；训练数据可能位于 `datasets/kangba_v44_train.jsonl` 等，请**先核实实际路径**。
- V4.4 已有大约 799 条训练/验证样本（710 训练 + 89 验证；不要把现有测试集并入训练）。
- 旧历史评测的结论：微调略增强抗吧风格，但上下文理解没有稳定变好；结构化群聊事实卡片仅带来较小改善。**这些结论不等于已证实 4B 一定更好。**
- 已有 31 道匿名群聊情境题，字段包含 `id`、`category`、`priority`、`context`、`incoming_message`、`evaluation_criteria`、`example_reply_for_human_review`。应优先查找：
  `datasets/QWbot_V44_真实群聊影子测试_匿名31题.jsonl`。
- **严格防数据泄漏**：喂给模型的只有 `context`、`incoming_message`，以及对四组一致的系统提示词；绝不把 `evaluation_criteria` 或 `example_reply_for_human_review` 放进模型输入。
- 旧群聊原始日志可能包含 QQ 号、用户名、NapCat WebUI token 等敏感信息；本阶段不得复制进新评测文件或提交至仓库，若发现泄漏应提醒用户轮换 token。只使用匿名化测试题。

## 3. 四模型分项目目录——必须保留隔离

实验根目录：`~/Qwen_Finetune/experiments/qwen_fourway/`。

```text
Qwen_Finetune/
├── .venv/                                   # 已有虚拟环境；不要重建
├── datasets/                                # 旧数据与匿名31题
├── outputs/                                 # V4.4 LoRA 与旧结果；禁止覆盖
└── experiments/
    └── qwen_fourway/
        ├── 01_1p7b_non_thinking/
        │   └── results/
        ├── 02_1p7b_thinking/
        │   └── results/
        ├── 03_4b_non_thinking/
        │   └── results/
        ├── 04_4b_thinking/
        │   └── results/
        ├── models/
        │   ├── 03_qwen3_4b_instruct_2507_bnb4/
        │   └── 04_qwen3_4b_thinking_2507_bnb4/
        ├── tools/
        │   ├── 00_check_env.py
        │   ├── 01_download_models.py
        │   └── 02_smoke_test.py
        └── README_先看.txt
```

| 编号 | 模型 | 权重源/位置 | 模式 | 说明 |
|---|---|---|---|---|
| **01** | Qwen3-1.7B | `unsloth/Qwen3-1.7B-unsloth-bnb-4bit` | `enable_thinking=False` | 基线；和02共用权重 |
| **02** | Qwen3-1.7B | 同上 | `enable_thinking=True` | 单独考察思考模式 |
| **03** | Qwen3-4B-Instruct-2507 | 本地 `models/03_qwen3_4b_instruct_2507_bnb4` | 固定非思考 | 对比容量提升 |
| **04** | Qwen3-4B-Thinking-2507 | 本地 `models/04_qwen3_4b_thinking_2507_bnb4` | 固定思考 | 对比专门的思考模型 |

03、04使用已下载的 Unsloth 4-bit 量化版。**2507 的 Instruct 与 Thinking 是两个独立模型，不是给 Instruct 简单传 `enable_thinking=True` 就能切换。** 所有四组初期都必须是**基础模型，不加载 V4.4 LoRA**。

## 4. 已发生的实际运行结果（来自用户日志）

之前共用的 `02_smoke_test.py` 曾在模型加载前调用：

```python
torch.cuda.reset_peak_memory_stats(0)
```

这在该 WSL/PyTorch 环境下报 `RuntimeError: Invalid device argument`。删除该早期调用后，02/03/04 均能执行 CUDA 模型加载与生成。**请检查当前脚本是否仍含此错误，不要重新引入。**

使用的简单测试题：

> 群友甲：昨天说不干了。群友乙：明天早到一个小时继续上班。群友丙：这是什么成分？请用一句自然的群聊口吻回复丙。

| 编号 | 运行状态 | GPU峰值显存（用户日志） | 生成耗时（用户日志） | 观察 |
|---|---|---:|---:|---|
| 01 | **尚未收到运行日志** | 未知 | 未知 | 必须补测 |
| 02 | 模型加载及最终回复生成成功 | 1.362 GB | 20.654 秒 | 有思考结束标记；回复“这事儿别问，干了就是。”，不够切题 |
| 03 | 模型加载及最终回复生成成功 | 3.376 GB | 11.044 秒 | 回复“别问是什么成分，就是早起打卡，跟闹钟斗智斗勇啊！”，逻辑一般 |
| 04 | **模型加载成功，但生成尚未完成** | 3.504 GB | 119.754 秒 | 原生成预算 1536 tokens；缺失 `</think>`，错误地将未完成思考当作最终回复 |

注意：上述显存是脚本报告的 `torch.cuda.max_memory_allocated`，不是 `nvidia-smi` 的设备总占用；上述推理时间不包括加载模型时间。不能把单题样本当作31题最终结论。

`02_smoke_test.py` 早期版本中，04默认 `max_new_tokens=1536`，并且若没有 `</think>` 就 `answer=raw`，从而误把整段推理输出为“最终回复”。上一轮提供过**修补版**，把04默认预算改为3072，并在缺失 `</think>` 时 `answer=''`、`response_ok=False`、记录 `hit_token_limit`。但**用户尚未反馈是否覆盖安装该修补版、也尚未提供复测结果**。Codex 必须以本机实际文件为准。

用户日志中的 Transformers `torch_dtype`/`max_length` 与 PyTorch Enum deprecation 警告不是这次失败主因；可在不改变模型行为前提下清理，但不得因警告盲目重装环境。

## 5. 当前工作重点 P0：保证04的推理边界正确

### 5.1 先审计，再修改

请打开 `experiments/qwen_fourway/tools/02_smoke_test.py`，核对以下问题：

- 04实际加载的权重路径与 tokenizer 是否匹配，模型配置是否为 `Qwen3-4B-Thinking-2507` 的量化版本。
- `apply_chat_template` 的输出和实际特殊标记是什么；不能假定所有 Qwen3 Thinking 版本格式完全相同。
- `max_new_tokens` 是否只是截断了尚未结束的思考；不要把它误诊成模型无法在 RTX 2070 上运行。
- 思考边界 `</think>`、可能的 `<think>` 及模型结束标记解析是否正确；优先以模型自身模板/官方文档和实际生成 token 为准。
- **在思考未结束时**：`answer` 必须为空，记录 `response_ok=false`、是否命中长度上限、实际生成 token 数与推理耗时，绝不能将分析文本作为 QQ 消息或者有效评测回答。
- 建议让 `smoke_result.json` 明确包含：`case`、`thinking`、`model`、`answer`、`response_ok`、`thinking_closed`、`hit_token_limit`、`prompt_tokens`、`generated_tokens`、`generate_seconds`、`load_seconds`、`peak_vram_gb`；不要将完整内部思考过程导出成聊天记录。
- 可用默认3072、备选4096尝试一次04；**必须可通过参数设置上限，不能无限生成**；耗时过长要诚实报告，不能强行宣称“思考一定更好”。
- 运行04期间在8GB卡上只加载一个模型，不要并发四份。

### 5.2 复测命令

```bash
cd ~/Qwen_Finetune
source .venv/bin/activate
python experiments/qwen_fourway/tools/02_smoke_test.py --case 01
python experiments/qwen_fourway/tools/02_smoke_test.py --case 02
python experiments/qwen_fourway/tools/02_smoke_test.py --case 03
python experiments/qwen_fourway/tools/02_smoke_test.py --case 04 --max-new-tokens 3072
```

逐个执行，确认前一个进程已经释放 GPU 再执行下一个。若04仍未生成最终回答，记录为 **incomplete**，允许继续评测其他组，不得用截断分析文本顶替结果。

## 6. 下一步工作 P1：31道题、四组、离线公平对照

### 6.1 输入与输出

- 输入：已有的匿名31题 JSONL。若不存在，请先报告缺失，检查之前测试工具包/数据目录，**不要按想象伪造31道题**。
- 样本中 `context` 为有序 `[{"speaker":"...","text":"..."}]`，`incoming_message` 为当前发言。
- 所有四组**沿用同一套系统提示词、上下文次序、当前问题、上下文截断策略**；不能为4B暗中加入更好的提示词后当作纯模型规模对比。
- 建议至少支持 `--check-only`、`--case 01|02|03|04|all`、`--limit 5`、`--limit 0`、`--repeats 2`、`--tag NAME`、`--resume`、`--max-new-tokens`，并有清晰的 `--help`。
- 推理预算需按思考/非思考合理区分，并单独统计**思考 token、最终答复 token、思考是否完成**。由于思考耗费额外 token，不应仅用相同 `max_new_tokens` 生硬判断胜负。
- 固定采样配置与每题随机种子，优先增加一个**确定性或低随机性**试验；不要把生成风格上的随机性当作模型能力差距。
- 训练集、评估标准和参考答复不允许进入推理输入。

### 6.2 输出隔离

建议新建（或在现有脚本设计上迭代）：

```text
experiments/qwen_fourway/
├── tools/
│   ├── 03_benchmark_real_chat.py       # 31题对照入口，具体命名可调整
│   └── 04_analyze_benchmark.py         # 统计与对比
├── 01_1p7b_non_thinking/results/bench_<tag>.jsonl
├── 02_1p7b_thinking/results/bench_<tag>.jsonl
├── 03_4b_non_thinking/results/bench_<tag>.jsonl
├── 04_4b_thinking/results/bench_<tag>.jsonl
└── reports/
    ├── four_model_benchmark_<tag>.csv
    └── four_model_benchmark_<tag>.md
```

每条推理至少记录：`question_id`、`repeat`、`case`、`model_identifier`、`thinking`、`context_present`、`prompt_tokens`、`generated_tokens`、`answer`、`response_ok`、`thinking_closed`（如适用）、`hit_token_limit`、`load_seconds`、`generate_seconds`、`peak_vram_gb`、`seed`、`error`。失败样本要保留状态行；要增量保存，避免04跑几个小时中途退出而丢失全部结果。

**评测必须区分**：31题中20题带历史消息，11题不带历史消息。总共31题×2轮×4组=**248条尝试**；如果先跑每组5题×1轮，则为20条尝试。不能把只有11题无上下文的差异称为“长记忆变好”。

### 6.3 评分维度

- **上下文/问题理解 0–2**：是否正确回答当前问题、能引用必要历史、能辨别冲突/自述与事实。
- **语言自然度 0–2**：像真实群友、少模板化和冗长解释。
- **抗吧风格 0–2**：在适当场景有梗而不强行套梗。
- **事实错误/不当建议风险**：`Y/N` 并说明（如赛事转会、图片不可见、医疗、劳动法）。
- **未完成思考**：单列 `incomplete`，不能给整段分析文本打“最终回复”的分数。
- **性能**：加载耗时、生成耗时、峰值显存及有效回答比例。思考模型是否适合实时群聊必须结合耗时判断。
- 参考答案只是可接受答复的一个例子，不能做逐字匹配。建议生成可人工盲评 CSV，模型版本暂以随机标签表示，评分后再解盲；若实现盲评映射，密钥文件单独保存。

重点关注：群友昵称“太神”、冲突解释“悲区”、长跨度的“二缸石榴”与 `2-16`、从植物大战僵尸切到 LOL、工作“不干了”与“明天早到”的回旋镖、`3.11` 与 `3.9` 的数字/版本歧义、没有图像输入却作出看图判断，以及电竞即时事实与劳动法问题。

### 6.4 公平性及技术注意

- 测试不应加载任何 V4.4 LoRA；本阶段隔离**模型规模和思考模式**的影响。
- 由于01/02是同一权重切模式、03/04是两套不同的2507模型，这不是严格的单变量全因子实验；报告中必须说明版本与训练差异，不能把所有改善唯一归因于“多2.3B参数”或“打开思考开关”。
- 04若非常慢，可先只测5题、记录真实速度，再决定是否运行全量；**不能因为04未完成就用它的推理片段与其他组最终回答作质量比较**。
- 显存不够时优先安全降低输入长度/分批串行；严禁默默切到CPU并宣称是RTX 2070上的 GPU 速度。
- 不要未经许可用联网搜索补全测试题中实时电竞事实（那会改变比较对象）；应评价模型有没有承认不确定和避免编造。

## 7. 工作边界与安全要求

1. 不触碰生产 NapCat/QQ 群自动发言链路；测试完全离线。
2. 不重新下载已经存在的数 GB 模型；除非先核实损坏并向用户说明。
3. 不自动 `pip install -U` 或升级 `torch`、`transformers`、`unsloth`；先报告版本、必要性和可能兼容风险。
4. 不覆盖 `outputs/kangba_v44_lora`、训练集、旧评测 CSV、用户现有 `results/smoke_result.json`；**当前冒烟脚本每次会覆写固定的 `smoke_result.json`，复测前必须先备份旧结果或改用带时间/标签的新输出文件名**。新评测用新 `tag`，代码修改前备份或使用版本控制。
5. 不把真实QQ号码、个人聊天原文或访问 token 输出到公开日志或提交仓库。
6. 不把模型的思考/推理文字发给群友；只在 `response_ok=true` 且最终回答可用时产出候选群聊消息。
7. 没有完成GPU验证时明确写“未运行/待验证”，不能凭静态检查冒称成功。
8. 任何会修改生产行为、启动长期常驻机器人、接入联网服务或开始训练的动作，必须先获得用户确认。

## 8. 建议执行顺序与验收标准

### Phase A — 只读审计（先完成）

- [ ] 确认当前在 `~/Qwen_Finetune`、正确的 `.venv`、WSL 和 RTX 2070。
- [ ] 检查四个项目和模型文件路径是否齐全；核实01/02共用基础权重。
- [ ] 确定 `02_smoke_test.py` 是否已应用04修复补丁；阅读当前逻辑，不盲目覆盖。
- [ ] 找到31题匿名JSONL，验证31个唯一ID及字段。
- [ ] 输出“当前状态、缺失项、需要修改哪些文件”的简短清单。

### Phase B — 最小修复 + 冒烟测试

- [ ] 修复/确认 CUDA 初始化与模型加载顺序。
- [ ] 修复/确认04的思考结束边界、截断状态和最终回复隔离。
- [ ] 补测01；复核02/03；04先3072 tokens 测试，必要时讨论4096及耗时。
- [ ] 每组 `results` 留存可核实JSON，报告显存、时延、有效回复。
- [ ] **特别验收**：04未出现 `</think>` 时 `answer` 为空、`response_ok=false`；永不把推理文本标成最终回复。

### Phase C — 小规模四模型评测

- [ ] 对31题中前5题分别运行四组模型，1轮，共20次尝试，保证每组输入语义一致。
- [ ] 检查结果路径、行数、去重/续跑行为、评分字段和泄漏防护。
- [ ] GPU时间与显存开销合理，再决定是否全量执行。

### Phase D — 完整评测与建议

- [ ] 用户确认后，31题×2轮×4组完成248条尝试；失败、截断单列。
- [ ] 生成可盲评 CSV、统计汇总及 `reports/*.md`。
- [ ] 针对“哪个模型理解好、哪个可实时部署、是否值得训练4B”提供基于结果的结论。
- [ ] 若04推理太慢或长时间无法完成，明确不适合当前实时群聊场景，不要为了完成表格强行算作有效回答。

## 9. Codex 最终回复必须包含

- 修改/新增文件列表（相对路径）及每个文件的用途。
- 验证过的测试命令、真实输出路径和结果；未测试项注明原因。
- 01/02/03/04 的完成状态、峰值显存、生成时延和是否得到最终回答。
- 明确回答04是否还会泄漏思考过程；有哪些剩余问题。
- 下一个最小可执行步骤，以及是否需要用户确认进行31题全量测试。

---

## 10. 可以直接给 Codex 的启动语

> 请先阅读根目录的 `CODEX_QWbot_四模型实验_项目交接与执行任务.md`，按其中 Phase A → Phase B 顺序检查项目。先不要训练模型、不要重装环境、不要连接 QQ 生产机器人。核对 `experiments/qwen_fourway/tools/02_smoke_test.py` 当前版本，重点确保04思考模型未生成 `</think>` 时不把分析文本当作群聊回复，然后补测01并验证四组的独立输出。完成后列出真实运行结果和下一步评测命令。除非我确认，否则不要执行耗时较长的248条完整评测。
