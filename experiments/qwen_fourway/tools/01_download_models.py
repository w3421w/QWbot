#!/usr/bin/env python3
"""按编号下载指定基础模型：1.7B共用HF缓存，两个4B分别存目录。"""
import argparse
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPECS = {
    '12': ('unsloth/Qwen3-1.7B-unsloth-bnb-4bit', None),
    '03': ('unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit', ROOT / 'models/03_qwen3_4b_instruct_2507_bnb4'),
    '04': ('unsloth/Qwen3-4B-Thinking-2507-unsloth-bnb-4bit', ROOT / 'models/04_qwen3_4b_thinking_2507_bnb4'),
}

def main():
    parser = argparse.ArgumentParser(description='独立下载Qwen四组实验的基础模型。')
    parser.add_argument('--case', choices=['01','02','03','04','all'], required=True)
    args = parser.parse_args()
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit('缺少 huggingface_hub，请激活原有 .venv 后运行。') from exc
    ids = ['12', '03', '04'] if args.case == 'all' else ['12' if args.case in ('01','02') else args.case]
    print('下载源:', os.environ.get('HF_ENDPOINT', '默认Hugging Face'))
    for ident in ids:
        repo, local = SPECS[ident]
        print('\n======== 开始下载', ident, repo, '========', flush=True)
        try:
            if local is None:
                path = snapshot_download(repo_id=repo)
                (ROOT / 'models/01_02_1p7b_共享权重缓存位置.txt').write_text(
                    f'两组1.7B实验共用权重缓存，不做重复复制。\nrepo: {repo}\ncache: {path}\n', encoding='utf-8'
                )
            else:
                local.mkdir(parents=True, exist_ok=True)
                path = snapshot_download(repo_id=repo, local_dir=str(local))
            print('[完成]', path, flush=True)
        except Exception as exc:
            print('[失败]', type(exc).__name__, str(exc), flush=True)
            raise SystemExit(1) from exc
    print('\n下载完成。下一步使用 02_smoke_test.py 测试每个模型。')

if __name__ == '__main__':
    main()
