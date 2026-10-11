抗压背锅吧 QWbot V4：训练与四模型对比工具
=============================================
适用环境：QWbot / Ubuntu WSL2 / ~/Qwen_Finetune / RTX 2070 8GB

文件结构（直接解压到 ~/Qwen_Finetune）：
  prepare_kangba_v4_kangba.py     语料提取与build
  train_kangba_v4.py              V4 QLoRA训练脚本
  compare_kangba_v4.py            原始Qwen、V2、V3、V4同题对比
  datasets/kangba_v4_kangba_candidates.csv      1000条已标记approve=y/n的候选（105条y）
  datasets/kangba_v4_kangba_seed_examples.csv   55条原创示范
  datasets/kangba_v4_kangba_holdout_prompts.txt 12条未用于V4训练的测试题

【推荐操作】
1. 在Windows解压ZIP到\\\\wsl$\\QWbot\\home\\alex\\Qwen_Finetune
   允许覆盖旧的 prepare_kangba_v4_kangba.py 和 datasets/kangba_v4_kangba_candidates.csv。
   不会触及旧的V2、V3 LoRA，也不会改动11.8万原始语料。

2. WSL：
  cd ~/Qwen_Finetune
  source .venv/bin/activate
  python prepare_kangba_v4_kangba.py build
  python train_kangba_v4.py --check-only

   正常预计出现：训练141条，验证19条（根据当前审阅CSV及默认随机种子）。
   若数字不同，请先检查文件是否为已经审核的CSV。

3. 开始训练：
  python train_kangba_v4.py

   默认3 epochs / 5e-5学习率 / LoRA rank 8 / FP16 / 4bit Qwen3 1.7B；
   每轮验证并保存，结束时恢复验证Loss最好的checkpoint并保存LoRA：
  outputs/kangba_v4_lora/adapter_model.safetensors
  outputs/kangba_v4_lora/training_report.json

   需要改实验参数（不推荐直接盲目加大）：
  python train_kangba_v4.py --epochs 2 --lr 3e-5

   保护已有V4：若输出已经存在，默认拒绝覆盖。
   确定需要重训且已备份时，才加 --overwrite。

4. 对比测试（如果V2/V3目录还在，会自动把它们纳入）：
  python compare_kangba_v4.py --limit 8
  python compare_kangba_v4.py --limit 0

   报告输出：
  outputs/kangba_v4_comparison.txt
  outputs/kangba_v4_comparison.json

【注意】
- V4仅学习抗吧表达风格，不以LOL为唯一话题。真实帖子仍可能因来源偏向电竞。
- 数据合计160条，其中105条论坛审核入选，55条单独编写示范；
  训练验证默认141/19。当前数量非常少，V4只是小样本实验。
- 已审核候选也可能受楼中楼缺失上下文或历史赛事真实性影响，不是事实知识库。
- train脚本仅对助手completion计算loss，并使用Qwen3非思考模式提示词。
- 为了可比性，compare脚本给所有模型使用相同V4系统提示词，采用固定采样种子；
  模型回复是抽样结果，不能只看一次测试。评估需要同时考虑相关性、短句、抗吧风格、事实错误。
- 提前终止 / CUDA OOM / 保存异常：请保留终端完整错误，继续排查。
