QWbot V4.4 — 抗吧式精准接梗专项增强（2026-10-08）
=======================================================
本工具包不会覆盖V4.3模型及旧数据。新增数据均为AI原创示范，并非真实贴吧评论。

【已有条件】
QWbot / Ubuntu WSL, ~/Qwen_Finetune
已安装Unsloth/PyTorch/TRL的.venv
本地已有 datasets/kangba_v43_selected_metadata.csv
本地已有 datasets/kangba_v43_train.jsonl、kangba_v43_valid.jsonl
本地已有 outputs/kangba_v43_lora（供训练后对比）

【文件】
prepare_kangba_v44.py             V4.4数据预审与构建（Python标准库）
train_kangba_v44.py               原始Qwen3 1.7B 4bit QLoRA训练，不继承V4.3
compare_kangba_v44.py             V4.3与V4.4的匿名同条件测试与评分汇总
v44_inputs/kangba_v44_authored_examples.csv  300条原创专项样本
v44_inputs/kangba_v44_holdout.csv           V4.4独立测试题；不参加训练
v44_inputs/kangba_v43_holdout.csv          历史测试题，构建时排除
v44_inputs/kangba_v41_new_holdout.csv       历史测试题，构建时排除
datasets/kangba_v44_holdout.csv             32条测试题，测试脚本读取

【第一步：拷贝解压】
在Windows资源管理器访问：
\\wsl$\QWbot\home\alex\Qwen_Finetune
将ZIP内容原样解压到上述Qwen_Finetune文件夹。
不要只复制Python脚本；v44_inputs子目录也要完整复制。

【第二步：数据预审和构建】
cd ~/Qwen_Finetune
source .venv/bin/activate
python prepare_kangba_v44.py prepare

会生成 datasets/kangba_v44_review.csv。
新样本默认approve=y。你可以先用表格软件打开，发现不想学习的回复时改为n。
注意这是原创示范，绝不是对300条真实贴吧数据的审核结果。

python prepare_kangba_v44.py build
python train_kangba_v44.py --check-only
python compare_kangba_v44.py --check-only

在当前用户此前汇报的501条V4.3数据上，隔离复测的结果为：
旧数据441训练+60验证；新增通过去重/留出保护的298条分配269训练+29验证；
最终710条训练、89条验证、合计799条（实际以你本机运行报告为准）。
5条短梗因原训练集已含相同答案而保留了新的上下文示范，
1条与旧验证集短梗答案重复被排除，另1条与旧测试题过于相似被排除。

【第三步：训练V4.4】
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME="$HOME/Qwen_Finetune/hf_cache"
python train_kangba_v44.py

训练使用与V4.3相同的基础Qwen3 1.7B、4bit、LoRA r=8、
默认3 Epoch、5e-5学习率、FP16、max_length=512，system提示词保持相同。
保存至outputs/kangba_v44_lora；checkpoint保存在outputs/kangba_v44_checkpoints。
不会覆盖outputs/kangba_v43_lora。
首次开始时一定注意“标签检查”，只有助手的回复应该参与Loss计算。
如果标签检查失败，停止并将完整报错发给助手，不要绕过检查。

【第四步：匿名对比V4.3与V4.4】
python compare_kangba_v44.py --limit 16 --repeats 2

自动生成：
outputs/kangba_v44_blind_scoring.csv
outputs/kangba_v44_blind_key.json
outputs/kangba_v44_raw_results.json

先审核blind_scoring的A/B回复，在which_is_better_A_B_tie填写A/B/Tie，
然后在relevance_A_0_2等列填写0~2；如有编造或严肃聊天失当，填写风险标记。
评分完成前不要看blind_key.json。
评分完成后：python compare_kangba_v44.py --summarize

【重要限制】
- 系统和训练流程没有在你的RTX2070上运行过；本工具包已经做的是无GPU格式检查。
- V4.4语料中原创示范比重较高，有可能提升拆台能力，也可能稀释真实抗吧口吻；必须用实际回复验证。
- 新增的32条独立题没有加入任何训练文件，后续也不要合并进训练集。
- 只保留原始501条中的训练/验证分配，不会洗牌重分。
- 审核表中approve=y表示本次允许训练，并不意味着这句话来自抗压背锅吧或事实已核实。
- 对事实查证、情绪困难场景设置硬性评估，不用平均风格分抵消。
