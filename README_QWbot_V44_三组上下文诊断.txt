QWbot｜V4.4 群聊上下文三组对照诊断（离线，无需重新训练）
=======================================================

目标
----
利用先前31道真实群聊匿名留出题，定位模型没有利用群聊上下文的根源：
BASE_RAW    原始Qwen3-1.7B + 原始群聊历史
V44_RAW     Qwen3-1.7B + V4.4 LoRA + 同一段原始群聊历史
V44_CARDS   Qwen3-1.7B + V4.4 LoRA + 原始群聊历史 + 自动提取的信息卡片

第三组信息卡片仅从题目自身的context生成。提取包括说话者、原话、称呼自述、数字和更正标记；没有连接外部知识库，不使用参考答案，也没有在QQ群长期记忆中存取任何数据。
这是一项“结构化输入”实验，不是证明已经完成持久群聊记忆系统。

操作前提
--------
1. Windows/WSL2 Ubuntu QWbot；RTX 2070 8GB。
2. ~/Qwen_Finetune/.venv 已安装Unsloth、Torch、PEFT等（之前V4.4训练测试成功的环境即可）。
3. ~/Qwen_Finetune/outputs/kangba_v44_lora/adapter_config.json 存在。
4. ZIP中脚本和 datasets/ 子文件夹整体解压到 ~/Qwen_Finetune，保留原路径。
5. 不需上传真实NapCat日志；工具包题目已经匿名化。

步骤一：检查（不启用GPU）
---------------------
cd ~/Qwen_Finetune
source .venv/bin/activate
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME="$HOME/Qwen_Finetune/hf_cache"
python compare_kangba_v44_context_triad.py --check-only

该命令应展示3种模式、31题（有上下文20题、无上下文11题）、前几题两种提示词预览以及“检查通过”。如模型已本地缓存，正式测试可以不依赖额外网络获取模型文件。

步骤二：先试5题
--------------
python compare_kangba_v44_context_triad.py --limit 5 --repeats 1 --tag trial

本步骤15次生成，对比“ 小悲 / 太神 / 悲区 / 二缸石榴 / LOL话题切换 ”。测试完成后可上传：
outputs/kangba_v44_context_triad_trial_blind_scoring.csv

步骤三：正式31题、每题两次
-----------------------
python compare_kangba_v44_context_triad.py --limit 0 --repeats 2 --tag full

预期：31题×2轮×3组=186次生成，输出62行匿名A/B/C对比。显存峰值与耗时以你机器实测为准。
注意：先完成trial再执行full，不要重用同一个tag。

如果中断，请追加--resume，其他选项完全相同：
python compare_kangba_v44_context_triad.py --limit 0 --repeats 2 --tag full --resume
脚本对每次结果即时落盘，续跑不会覆盖已生成的模式。

步骤四：上传匿名结果
------------------
请只上传：
outputs/kangba_v44_context_triad_full_blind_scoring.csv

先不要上传或者打开：
outputs/kangba_v44_context_triad_full_blind_key.json （A/B/C身份映射，仅解盲后使用）
outputs/kangba_v44_context_triad_full_raw_long.csv （带身份的调试数据）

盲评CSV留有评审列，您可以直接上传未填写的文件，由助手基于问题、历史、三条答案分别评分。等评分完成后再上传blind_key.json。不要把“评价标准”和“参考回复”当成模型的输入或训练样本。

输出文件说明
------------
outputs/kangba_v44_context_triad_<tag>_blind_scoring.csv  匿名A/B/C评分表（请上传）
outputs/kangba_v44_context_triad_<tag>_blind_key.json     解盲映射（评分前保密）
outputs/kangba_v44_context_triad_<tag>_raw_long.csv       三组逐条输出（仅内部调试）
outputs/kangba_v44_context_triad_<tag>_metadata.json       运行配置、采样参数、系统提示词哈希

可选：评分后本地汇总
python compare_kangba_v44_context_triad.py --summarize --tag full
（不使用GPU，但评分文件必须先填写。生成 ..._summary.json）

三种结果怎么判断
--------------
1. BASE_RAW 明显好于 V44_RAW：优先检查LoRA样本质量、训练标签/分布、是否出现基础能力退化。
2. V44_CARDS 明显好于 V44_RAW，尤其在20条有上下文的题中：优先研发群聊事实提取、引用/昵称/冲突标注与上下文组织。
3. 三者都差：需要检查1.7B模型能力、问题难度、提示词长度/训练样本质量；不宜把锅全部归于记忆系统。
4. 无上下文11题的V44_RAW与V44_CARDS理论上获得同样提示文本，不能把它们的差异解读成群聊事实提取的增益。

重要边界
--------
- 31题是匿名化、改写后的留出题，不等同于原群原样的数据分布；不要放进V4.5训练。
- 所有三组使用同一套系统提示词与参数；BASE_RAW通过PEFT的disable_adapter()暂时关闭LoRA。
- 结构化卡片组会在原文之外增加卡片和少量解释，因此提升不能单独归因于“持久记忆”。
- 模型不接入NapCat、QQ发送接口，不修改LoRA权重，不读取本地群成员私有数据库，不做事实联网检索。
- 医疗、法律、实时赛事需要单独的事实源/专业处理链，不能用这套实验代替。
- 若运行出错请复制完整错误信息、命令和GPU信息，不建议删除标签检查或绕过异常。
