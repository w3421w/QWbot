#!/usr/bin/env python3
"""依次运行单个模型的量化推理冒烟测试；不会写入QQ，也不会修改LoRA。"""
import argparse
import gc
import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CASE_INFO = {
    '01': ('01_1p7b_non_thinking', 'unsloth/Qwen3-1.7B-unsloth-bnb-4bit', False),
    '02': ('02_1p7b_thinking', 'unsloth/Qwen3-1.7B-unsloth-bnb-4bit', True),
    '03': ('03_4b_non_thinking', str(ROOT / 'models/03_qwen3_4b_instruct_2507_bnb4'), False),
    '04': ('04_4b_thinking', str(ROOT / 'models/04_qwen3_4b_thinking_2507_bnb4'), True),
}
DEFAULT_PROMPT = '群友甲：昨天说不干了。群友乙：明天早到一个小时继续上班。群友丙：这是什么成分？请用一句自然的群聊口吻回复丙。'

def main():
    ap = argparse.ArgumentParser(description='Qwen四组模型：逐个本地冒烟测试（只跑一组）。')
    ap.add_argument('--case', required=True, choices=sorted(CASE_INFO))
    ap.add_argument('--prompt', default=DEFAULT_PROMPT)
    ap.add_argument('--max-new-tokens', type=int, default=0,
                    help='不填时：非思考192，1.7B思考768，4B思考3072；思考可能耗时数分钟')
    ap.add_argument('--tag',
                    help='结果文件标签，保存为对应模型目录/results/smoke_<tag>.json')
    args = ap.parse_args()
    target_dir, source, thinking = CASE_INFO[args.case]
    if args.tag is not None and not re.fullmatch(r'[A-Za-z0-9._-]+', args.tag):
        raise SystemExit('--tag只能包含字母、数字、点、下划线和连字符，请更换tag')
    result_name = f'smoke_{args.tag}.json' if args.tag is not None else 'smoke_result.json'
    out_dir = ROOT / target_dir / 'results'
    out_path = out_dir / result_name
    if out_path.exists():
        raise SystemExit(f'结果文件已存在，不会覆盖：{out_path}\n请更换tag后重试')
    if args.case in ('03','04') and not (Path(source) / 'config.json').is_file():
        raise SystemExit(f'尚未下载模型: {source}\n先执行 python tools/01_download_models.py --case {args.case}')
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit('缺少torch/transformers，请先检查环境：00_check_env.py') from exc
    if not torch.cuda.is_available():
        raise SystemExit('CUDA不可用。请在WSL中先运行 nvidia-smi 并确认使用正确的 .venv')
    # 每个case在独立Python进程运行，CUDA峰值计数从零开始。
    # 不能在CUDA上下文尚未初始化时重置，否则部分PyTorch/WSL组合报 Invalid device argument。
    print('=== 开始加载 ===', flush=True)
    print('实验:', args.case, target_dir, flush=True)
    print('来源:', source, flush=True)
    print('模式:', '思考' if thinking else '非思考', flush=True)
    print('模型均为基础模型，不加载V4.4 LoRA。', flush=True)
    start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=False)
    # 不用device_map="auto"，防止8GB显卡不够时悄悄跑到CPU导致比较失真。
    model = AutoModelForCausalLM.from_pretrained(
        source, torch_dtype=torch.float16, device_map={'': 'cuda:0'},
        low_cpu_mem_usage=True, trust_remote_code=False,
    )
    model.eval()
    # 模型加载完成后CUDA上下文已建立；不重置峰值，保留包括模型加载在内的真实峰值。
    print('CUDA设备:', torch.cuda.get_device_name(torch.cuda.current_device()), flush=True)
    load_secs = time.perf_counter() - start
    messages = [
        {'role': 'system', 'content': '你是QQ群里的普通群友。先理解聊天内容，再简短、自然地回答最后的问题。'},
        {'role': 'user', 'content': args.prompt},
    ]
    # 旧版Qwen3 1.7B可切换；2507新模型仅支持各自固定的思考/非思考模式。
    template_kwargs = {'tokenize': False, 'add_generation_prompt': True}
    if args.case in ('01', '02'):
        template_kwargs['enable_thinking'] = thinking
    text = tokenizer.apply_chat_template(messages, **template_kwargs)
    encoded = tokenizer(text, return_tensors='pt').to('cuda:0')
    max_new = args.max_new_tokens or (3072 if args.case == '04' else 768 if thinking else 192)
    if max_new < 1:
        raise SystemExit('--max-new-tokens必须为正整数')
    params = {
        'max_new_tokens': max_new,
        'do_sample': True,
        'temperature': 0.6 if thinking else 0.7,
        'top_p': 0.95 if thinking else 0.8,
        'top_k': 20,
        'pad_token_id': tokenizer.eos_token_id,
    }
    print('开始生成。思考模式可能耗时较久……', flush=True)
    t0 = time.perf_counter()
    with torch.inference_mode():
        gen = model.generate(**encoded, **params)
    torch.cuda.synchronize()
    secs = time.perf_counter() - t0
    gen_ids = gen[0][encoded['input_ids'].shape[-1]:]
    raw = tokenizer.decode(gen_ids, skip_special_tokens=False)
    # Thinking-2507 通常在输出结束时出现 </think>，可能没有开头的 <think>。
    # 如果尚未出现 </think>，说明目前只有思考内容：绝不能当作最终群聊回复。
    thinking_closed = '</think>' in raw
    if thinking:
        if thinking_closed:
            reasoning, answer = raw.rsplit('</think>', 1)
            reasoning = reasoning.replace('<think>', '').strip()
        else:
            reasoning, answer = raw.replace('<think>', '').strip(), ''
    else:
        reasoning, answer = '', raw
    for sp in ('<|im_end|>', '<|endoftext|>', '<|fim_suffix|>'):
        answer = answer.replace(sp, '')
    answer = answer.strip()
    hit_token_limit = int(gen_ids.shape[-1]) >= max_new
    response_ok = bool(answer) and (not thinking or thinking_closed)
    if response_ok:
        print('\n=== 最终回复 ===\n' + answer, flush=True)
    else:
        print('\n=== 最终回复 ===\n[未完成：思考仍在进行，尚未产生可发送给QQ群的最终回复]', flush=True)
    if thinking:
        print('\n=== 推理生成情况 ===')
        print('推理文本字符数:', len(reasoning))
        print('是否生成思考结束标记:', thinking_closed)
        print('达到生成Token上限:', hit_token_limit)
        if not response_ok:
            print('建议：可尝试 --max-new-tokens 4096；约需数分钟，仍不保证模型完成思考。')
    stats = {
        'case': args.case, 'model': source, 'thinking': thinking,
        'prompt': args.prompt, 'answer': answer,
        'reasoning_chars': len(reasoning),
        'thinking_closed': thinking_closed,
        'response_ok': response_ok, 'hit_token_limit': hit_token_limit,
        'load_seconds': round(load_secs, 3), 'generate_seconds': round(secs, 3),
        'prompt_tokens': int(encoded['input_ids'].shape[-1]),
        'generated_tokens': int(gen_ids.shape[-1]),
        'peak_vram_gb': round(torch.cuda.max_memory_allocated(0) / 1024**3, 3),
        'model_base_only': True,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        with out_path.open('x', encoding='utf-8') as result_file:
            json.dump(stats, result_file, ensure_ascii=False, indent=2)
            result_file.write('\n')
    except FileExistsError as exc:
        raise SystemExit(f'结果文件已存在，不会覆盖：{out_path}\n请更换tag后重试') from exc
    print('\n结果文件:', out_path)
    print('显存峰值(GB):', stats['peak_vram_gb'], '生成耗时(秒):', stats['generate_seconds'])

if __name__ == '__main__':
    main()
