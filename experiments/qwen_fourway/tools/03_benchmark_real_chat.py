#!/usr/bin/env python3
"""四个基础模型对真实群聊匿名题的串行对照测试。

本脚本不加载 LoRA、不连接 QQ/NapCat；每条结果独立追加到对应模型 results 目录。
"""
import argparse
import gc
import hashlib
import json
import os
import random
import re
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT.parents[1] / 'datasets' / 'QWbot_V44_真实群聊影子测试_匿名31题.jsonl'
CASE_INFO = {
    '01': ('01_1p7b_non_thinking', 'unsloth/Qwen3-1.7B-unsloth-bnb-4bit', False),
    '02': ('02_1p7b_thinking', 'unsloth/Qwen3-1.7B-unsloth-bnb-4bit', True),
    '03': ('03_4b_non_thinking', str(ROOT / 'models/03_qwen3_4b_instruct_2507_bnb4'), False),
    '04': ('04_4b_thinking', str(ROOT / 'models/04_qwen3_4b_thinking_2507_bnb4'), True),
}
SYSTEM_PROMPT = '你是QQ群里的普通群友。先理解聊天内容，再简短、自然地回答最后的问题。'
END_MARKERS = ('<|im_end|>', '<|endoftext|>', '<|fim_suffix|>')
REQUIRED_FIELDS = ('question_id', 'case', 'repeat', 'answer', 'response_ok', 'generated_tokens',
                   'prompt_tokens', 'generate_seconds', 'peak_vram_gb', 'seed', 'error')


def parse_args():
    ap = argparse.ArgumentParser(description='QWbot真实群聊31题四模型串行对照测试（基础模型，无LoRA）。')
    ap.add_argument('--check-only', action='store_true', help='只校验数据、路径、参数和提示词隔离，不加载GPU模型')
    ap.add_argument('--case', default='all', choices=('01', '02', '03', '04', 'all'))
    ap.add_argument('--limit', type=int, default=0, help='题目数量；0为全部31题')
    ap.add_argument('--repeats', type=int, default=1, help='每题生成次数，必须为正整数')
    ap.add_argument('--tag', required=False, help='结果文件标签，只允许字母、数字、点、下划线和连字符')
    ap.add_argument('--resume', action='store_true', help='跳过已有有效记录，不重复写入同一题次')
    ap.add_argument('--seed', type=int, default=20261009, help='可复现随机种子基数')
    ap.add_argument('--max-new-tokens', type=int, default=0, help='统一生成上限；0使用各模式默认值')
    ap.add_argument('--thinking-max-new-tokens', type=int, default=0, help='思考模型独立生成上限；0使用默认值')
    return ap.parse_args()


def load_cases(path):
    if not path.is_file():
        raise ValueError(f'测试数据不存在: {path}')
    rows = []
    with path.open(encoding='utf-8') as fh:
        for line_no, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f'第{line_no}行不是有效JSON: {exc}') from exc
            for key in ('id', 'context', 'incoming_message'):
                if key not in row:
                    raise ValueError(f'第{line_no}行缺少字段: {key}')
            if not isinstance(row['id'], str) or not row['id']:
                raise ValueError(f'第{line_no}行id无效')
            if not isinstance(row['context'], list) or not all(isinstance(x, dict) for x in row['context']):
                raise ValueError(f'第{line_no}行context必须是对象数组')
            for item in row['context']:
                if not isinstance(item.get('speaker'), str) or not isinstance(item.get('text'), str):
                    raise ValueError(f'第{line_no}行context项必须有speaker/text字符串')
            if not isinstance(row['incoming_message'], str):
                raise ValueError(f'第{line_no}行incoming_message必须是字符串')
            # 这些字段可存在于数据，但只能在校验层读取，绝不传给模型。
            rows.append(row)
    if len(rows) != 31:
        raise ValueError(f'测试题数量应为31，实际为{len(rows)}')
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('测试题id存在重复')
    return rows


def build_messages(row):
    """只使用 context 和 incoming_message 构造固定格式提示词。"""
    history = '\n'.join(f"{item['speaker']}：{item['text']}" for item in row['context'])
    user_text = f'聊天记录：\n{history}\n\n当前问题：\n{row["incoming_message"]}'
    return [{'role': 'system', 'content': SYSTEM_PROMPT}, {'role': 'user', 'content': user_text}]


def seed_for(base, question_id, repeat):
    digest = hashlib.sha256(f'{base}:{question_id}:{repeat}'.encode()).digest()
    return int.from_bytes(digest[:4], 'big')


def valid_record(obj, case, question_id, repeat):
    return (isinstance(obj, dict) and obj.get('case') == case and obj.get('question_id') == question_id
            and obj.get('repeat') == repeat and all(k in obj for k in REQUIRED_FIELDS)
            and isinstance(obj.get('response_ok'), bool))


def read_existing(path, case):
    seen = set()
    if not path.exists():
        return seen
    with path.open(encoding='utf-8') as fh:
        for line_no, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = (obj.get('question_id'), obj.get('repeat'))
            if isinstance(key[0], str) and isinstance(key[1], int) and valid_record(obj, case, key[0], key[1]):
                seen.add(key)
    return seen


def output_path(case, tag):
    target_dir = ROOT / CASE_INFO[case][0] / 'results'
    return target_dir / f'benchmark_real_chat_{tag}.jsonl'


def validate(args, rows):
    if args.limit < 0:
        raise ValueError('--limit必须为非负整数')
    if args.repeats < 1:
        raise ValueError('--repeats必须为正整数')
    if args.max_new_tokens < 0 or args.thinking_max_new_tokens < 0:
        raise ValueError('生成Token上限必须为非负整数')
    if not isinstance(args.seed, int):
        raise ValueError('--seed必须为整数')
    if not args.tag:
        raise ValueError('必须提供--tag，以确保结果文件带标签且不会覆盖旧结果')
    if not re.fullmatch(r'[A-Za-z0-9._-]+', args.tag):
        raise ValueError('--tag只能包含字母、数字、点、下划线和连字符')
    selected = list(CASE_INFO) if args.case == 'all' else [args.case]
    selected_rows = rows if args.limit == 0 else rows[:args.limit]
    if args.limit > len(rows):
        raise ValueError(f'--limit不能大于{len(rows)}')
    for case in selected:
        target_dir, source, _ = CASE_INFO[case]
        if case in ('03', '04') and not (Path(source) / 'config.json').is_file():
            raise ValueError(f'尚未下载模型: {source}')
        out = output_path(case, args.tag)
        if out.exists() and not args.resume:
            raise ValueError(f'结果文件已存在，不会覆盖: {out}；请换tag或使用--resume')
    # 检查隔离：敏感评测字段不出现在任何推理消息中。
    forbidden = ('evaluation_criteria', 'example_reply_for_human_review')
    for row in selected_rows:
        serialized = json.dumps(build_messages(row), ensure_ascii=False)
        if any(word in serialized for word in forbidden):
            raise ValueError(f'提示词隔离失败: {row["id"]}')
    return selected, selected_rows


def parse_output(raw, thinking):
    thinking_closed = '</think>' in raw
    if thinking:
        if thinking_closed:
            reasoning, answer = raw.rsplit('</think>', 1)
            reasoning = reasoning.replace('<think>', '').strip()
        else:
            reasoning, answer = raw.replace('<think>', '').strip(), ''
    else:
        reasoning, answer = '', raw
    for marker in END_MARKERS:
        answer = answer.replace(marker, '')
    answer = answer.strip()
    return reasoning, answer, thinking_closed


def run_case(case, rows, args):
    target_dir, source, thinking = CASE_INFO[case]
    out = output_path(case, args.tag)
    out.parent.mkdir(parents=True, exist_ok=True)
    seen = read_existing(out, case) if args.resume else set()
    pending = [(row, repeat) for row in rows for repeat in range(1, args.repeats + 1)
               if (row['id'], repeat) not in seen]
    if not pending:
        print(f'[{case}] 无待测试题次，跳过: {out}', flush=True)
        return
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit('缺少torch/transformers，请先检查环境：00_check_env.py') from exc
    if not torch.cuda.is_available():
        raise SystemExit('CUDA不可用；按要求不会自动切换CPU')
    model = tokenizer = None
    try:
        print(f'[{case}] 加载 {source}', flush=True)
        tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=False, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(source, torch_dtype=torch.float16,
            device_map={'': 'cuda:0'}, low_cpu_mem_usage=True, trust_remote_code=False,
            local_files_only=True)
        model.eval()
        for row, repeat in pending:
            seed = seed_for(args.seed, row['id'], repeat)
            random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            messages = build_messages(row)
            template_kwargs = {'tokenize': False, 'add_generation_prompt': True}
            if case in ('01', '02'):
                template_kwargs['enable_thinking'] = thinking
            started = time.perf_counter()
            error = ''
            answer = ''
            reasoning = ''
            thinking_closed = False
            response_ok = False
            hit_token_limit = False
            prompt_tokens = generated_tokens = 0
            peak_vram = 0.0
            try:
                text = tokenizer.apply_chat_template(messages, **template_kwargs)
                encoded = tokenizer(text, return_tensors='pt').to('cuda:0')
                max_new = (args.thinking_max_new_tokens if thinking else args.max_new_tokens) or (3072 if case == '04' else 768 if thinking else 192)
                params = {'max_new_tokens': max_new, 'do_sample': True,
                          'temperature': 0.6 if thinking else 0.7, 'top_p': 0.95 if thinking else 0.8,
                          'top_k': 20, 'pad_token_id': tokenizer.eos_token_id}
                with torch.inference_mode():
                    gen = model.generate(**encoded, **params)
                torch.cuda.synchronize()
                gen_ids = gen[0][encoded['input_ids'].shape[-1]:]
                raw = tokenizer.decode(gen_ids, skip_special_tokens=False)
                reasoning, answer, thinking_closed = parse_output(raw, thinking)
                prompt_tokens = int(encoded['input_ids'].shape[-1])
                generated_tokens = int(gen_ids.shape[-1])
                hit_token_limit = generated_tokens >= max_new
                response_ok = bool(answer) and (not thinking or thinking_closed)
                if not response_ok and not error:
                    error = 'incomplete: 未产生最终回复' if thinking else 'empty_answer'
            except Exception as exc:  # 单题错误必须落盘后继续
                error = f'{type(exc).__name__}: {exc}'
            peak_vram = round(torch.cuda.max_memory_allocated(0) / 1024**3, 3)
            record = {'question_id': row['id'], 'case': case, 'repeat': repeat, 'answer': answer,
                      'response_ok': response_ok, 'reasoning_chars': len(reasoning),
                      'thinking_closed': thinking_closed if thinking else True,
                      'hit_token_limit': hit_token_limit, 'generated_tokens': generated_tokens,
                      'prompt_tokens': prompt_tokens, 'generate_seconds': round(time.perf_counter() - started, 3),
                      'peak_vram_gb': peak_vram, 'seed': seed, 'error': error,
                      'model': source, 'thinking': thinking, 'model_base_only': True}
            with out.open('a', encoding='utf-8') as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + '\n')
                fh.flush()
                os.fsync(fh.fileno())
            print(f'[{case}] {row["id"]} repeat={repeat} ok={response_ok}', flush=True)
    finally:
        del model, tokenizer
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


def main():
    args = parse_args()
    try:
        rows = load_cases(DATA_PATH)
        cases, selected_rows = validate(args, rows)
    except ValueError as exc:
        raise SystemExit(str(exc))
    print(f'数据校验通过: {DATA_PATH} ({len(rows)}题); 本次{len(selected_rows)}题, case={",".join(cases)}', flush=True)
    if args.check_only:
        print('check-only通过：未加载GPU模型，评测字段未进入模型提示词。')
        return
    for case in cases:
        run_case(case, selected_rows, args)


if __name__ == '__main__':
    main()
