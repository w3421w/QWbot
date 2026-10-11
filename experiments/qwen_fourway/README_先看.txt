QWbot Qwen3 1.7B / 4B 四组模型安装与隔离实验
==============================================

【目标】
01：1.7B 原始非思考（旧版Qwen3同一底座）
02：1.7B 原始思考（与01共用同一份模型权重）
03：4B-Instruct-2507 非思考，Unsloth 4-bit 预量化
04：4B-Thinking-2507 思考，Unsloth 4-bit 预量化
只安装和推理，不训练，不动旧V4.4 LoRA，不向QQ群发送消息。

【解压位置】
将ZIP内 experiments/ 文件夹整体放到：
/home/alex/Qwen_Finetune/
最终看到：
/home/alex/Qwen_Finetune/experiments/qwen_fourway/README_先看.txt

【共用的原有环境】
cd ~/Qwen_Finetune
source .venv/bin/activate
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME="$HOME/Qwen_Finetune/hf_cache"

【步骤1：先检查环境】
python experiments/qwen_fourway/tools/00_check_env.py
重点检查CUDA=True、Transformers >= 4.51、bitsandbytes存在。
如果Transformers版本较旧：先备份或单独建虚拟环境，不要直接破坏现有能训练的Unsloth环境。

【步骤2：下载，不用一次全部下完】
# 01/02: 两组共用一份1.7B基础权重，原来训练时通常已有缓存
python experiments/qwen_fourway/tools/01_download_models.py --case 01

# 03: 单独下载非思考4B（预量化，约2.6GB）
python experiments/qwen_fourway/tools/01_download_models.py --case 03

# 04: 单独下载思考4B（预量化，约2.7GB）
python experiments/qwen_fourway/tools/01_download_models.py --case 04

# 也可以一次性完成上述三份权重的安装
python experiments/qwen_fourway/tools/01_download_models.py --case all

下载中断可重复执行同一条命令，正常情况下会复用已下载的文件。

【步骤3：每个项目单独运行，不要同时占用GPU】
python experiments/qwen_fourway/tools/02_smoke_test.py --case 01
python experiments/qwen_fourway/tools/02_smoke_test.py --case 02
python experiments/qwen_fourway/tools/02_smoke_test.py --case 03
python experiments/qwen_fourway/tools/02_smoke_test.py --case 04

# 如果04一直在思考，给更高的生成Token预算：
python experiments/qwen_fourway/tools/02_smoke_test.py --case 04 --max-new-tokens 2048

【产物目录严格分开】
experiments/qwen_fourway/01_1p7b_non_thinking/results/smoke_result.json
experiments/qwen_fourway/02_1p7b_thinking/results/smoke_result.json
experiments/qwen_fourway/03_4b_non_thinking/results/smoke_result.json
experiments/qwen_fourway/04_4b_thinking/results/smoke_result.json

【重要事项】
1. 两种4B-2507模型是独立权重，不是同一个权重用enable_thinking开关切换。
2. 1.7B两组的权重相同，只有enable_thinking不同。
3. RTX2070不支持BF16，按FP16计算+预量化4bit权重运行；未实测用户GPU，可能发生OOM。
4. 思考模式可输出大量Token；脚本只打印最终回复、统计思考字符数。
5. 这些只是基础模型的冒烟测试，不代表正式31道题对照评分。
6. 训练与验证文件、旧V4.4 LoRA完全不修改；输出独立保存。
7. 模型可能下载到本地HF缓存（01/02），不要手动删除hf_cache。
8. 如果模型下载或加载报错，把完整报错和00_check_env.py输出发来。
