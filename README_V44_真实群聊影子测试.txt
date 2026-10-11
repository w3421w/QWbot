QWbot：V4.4群聊影子测试（31条，匿名）
================================

用途：仅在本地Qwen3-1.7B+V4.4 LoRA上生成31条基于群聊日志的模拟回复；不重新训练、不连接NapCat、不会发QQ群消息。

1. 将ZIP内脚本放到 ~/Qwen_Finetune/test_kangba_v44_real_chat.py
   将JSONL放到 ~/Qwen_Finetune/datasets/QWbot_V44_真实群聊影子测试_匿名31题.jsonl
   ZIP自身已按目录排好，直接保留目录结构解压到项目根目录即可。

2. 在QWbot的WSL Ubuntu终端：
   cd ~/Qwen_Finetune
   source .venv/bin/activate
   export HF_ENDPOINT=https://hf-mirror.com
   export HF_HOME="$HOME/Qwen_Finetune/hf_cache"
   python test_kangba_v44_real_chat.py --check-only

3. 先抽测5道：
   python test_kangba_v44_real_chat.py --limit 5 --tag quick

4. 再正式跑全部31道，每题可生成1或2次：
   python test_kangba_v44_real_chat.py --limit 0 --repeats 2 --tag full

   结果写在 outputs/kangba_v44_real_chat_full_scoring.csv
   然后上传这个CSV给ChatGPT评分、分析问题、规划真实群聊上下文与记忆机制。

注意：不把example_reply_for_human_review作为模型输入；它只供人工参考。
只给模型context与incoming_message。系统提示词从本机V4.4训练集读取。
重复运行须使用不同的--tag以免覆盖先前结果。
本文件和JSONL都不包含日志中的QQ号和NapCat令牌。
原始NapCat日志中曾包含WebUI令牌，务必轮换；真实群聊日志的长期存储/训练须有合适的知情同意。
