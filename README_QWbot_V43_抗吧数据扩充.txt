QWbot V4.3：抗压背锅吧语料专项扩充与审核
==============================================================
目标：修复V4.2评测中的接话不切题、机械拼接黑话、缺少反串/回旋镖/粉丝拆台能力。
要求：尽量使用真正经过审核的上下文匹配问答；不要只追求原始语料数量。
系统：WSL2 QWbot / Ubuntu / alex / ~/Qwen_Finetune / RTX2070 8GB。

一、工具包内容
  prepare_kangba_v43.py          完整的数据准备、扩充、审核合并、训练集构建脚本；纯Python标准库。
  train_kangba_v43.py            与现有V4训练环境相容的QLoRA训练脚本，路径独立。
  v43_inputs/                    经过审核的旧105条、旧55条原创示范、180条复审候选、
                                 144条新原创示范，以及旧42条与新40条测试题。
  预览_审核队列.csv              预先生成的324条审核队列预览；不是已经全部审核通过。
  预览_基线统计.json             本地流程测试得到的304条候选训练总量统计。

二、如何安装
解压ZIP中的 prepare_kangba_v43.py、train_kangba_v43.py 和整个 v43_inputs/
到：\\wsl$\QWbot\home\alex\Qwen_Finetune\
可直接保留已有的datasets文件夹以及V2、V3、V4所有模型。

三、第一步：生成新的审核表（只需运行一次）
  cd ~/Qwen_Finetune
  source .venv/bin/activate
  python prepare_kangba_v43.py prepare
输出：datasets/kangba_v43_review_queue.csv
包含：144条由助手原创、标记y的示范；180条旧论坛暂缓记录，approve为空。
如不喜欢某条原创示范，改approve为n；如审核某条论坛回复确认上下文准确，可改为y。
重要：论坛内容来源、楼中楼上下文和事实未经外部逐条核验，不能仅凭自动分数批准。
不要重新运行prepare以免覆盖你的审核进度；脚本默认禁止覆盖已存在结果。

四、第二步：继续从10.6万条原始训练候选提取800条全新候选
  python prepare_kangba_v43.py expand --limit 800
前提：原来的 datasets/kangba_train.jsonl 仍然存在于QWbot环境。
输出：datasets/kangba_v43_new_forum_candidates.csv
默认approve全部留空，不参与训练。自动筛选并不能保证父评论配对正确。
你可以把这个CSV上传到聊天，我再帮你逐条筛选、保留真实抗吧表达。
只有明确审核过的行改为approve=y后才会加入构建。

五、第三步：只构建经过明确批准的数据
  python prepare_kangba_v43.py build
输出：
  datasets/kangba_v43_train.jsonl
  datasets/kangba_v43_valid.jsonl
  datasets/kangba_v43_selected_metadata.csv
  datasets/kangba_v43_report.json
相同原帖、相同问句或相同回复不会跨训练集/验证集。
42条旧测试题、40条新的独立测试题以及之前的8道经典测试题不允许进入训练。
注意：字面相似过滤仍不能完全排除语义近重复，需要抽查。

六、可先检查格式，但暂时不必训练
  python train_kangba_v43.py --check-only
完成审核后，建议高质量训练样本增加到500条以上再做新一轮QLoRA微调。
如果确认要做304条的早期对照，才执行：
  export HF_ENDPOINT=https://hf-mirror.com
  export HF_HOME="$HOME/Qwen_Finetune/hf_cache"
  python train_kangba_v43.py
V4.3模型保存在outputs/kangba_v43_lora，完全不覆盖V4。

七、来源与可靠性
  旧105条：原论坛内容，已在前轮筛选过，事实与原始楼中楼关系未独立核验。
  旧55条：助手撰写的原创群聊示范。
  新144条：助手新撰写的原创群聊示范，origin=authored_example_v43。
  待复审180条：此前暂缓的论坛原句，默认不批准。
  扩充800条：从QWbot本机原始候选提取的未经审核新语料，默认不批准。
  40条V4.3测试题：原创独立评估材料，禁止进入训练。

八、质量重点
  A. 先判断群友在说什么：不是依靠出现“典”“闹麻了”就加分。
  B. 允许“典”“绷”等1-3字回复，只要上下文正确。
  C. 找出“双标/赛前赛后矛盾/过度吹捧”的笑点，用一句话拆台。
  D. 严肃求助、悲伤、真实新闻/转会，不强行整活也不编造。
  E. 训练和推理保持Qwen3非思考模式，验证LoRA是否生效。

九、本地测试说明
在此前105条+55条+新144条的本地输入上：训练268条、验证36条，
共304条。检测：训练/验证不共享相同问句或相同回复；
新论坛扩充候选不会未经批准自动进入训练集；V4.3训练脚本的check-only通过。
这不是在你的GPU环境完成的新模型训练，也不意味着304条已经达到最终效果。
