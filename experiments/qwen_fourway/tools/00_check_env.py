#!/usr/bin/env python3
"""环境和项目布局检查；不会下载或运行GPU推理。"""
import importlib.metadata as md
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]
print('=== QWbot 四组模型实验：环境检查 ===')
print('Python:', sys.version.split()[0])
print('实验目录:', ROOT)
print('老项目目录:', PROJECT)
print('当前Python:', sys.executable)
print('HF_ENDPOINT:', os.environ.get('HF_ENDPOINT', '(未设置)'))
print('HF_HOME:', os.environ.get('HF_HOME', '(未设置，使用默认缓存)'))
for pkg in ('torch', 'transformers', 'accelerate', 'bitsandbytes', 'huggingface-hub', 'safetensors', 'unsloth'):
    try:
        version = md.version(pkg)
        print(f'[OK] {pkg}: {version}')
    except md.PackageNotFoundError:
        print(f'[缺少] {pkg}')
try:
    import torch
    print('CUDA可用:', torch.cuda.is_available())
    if torch.cuda.is_available():
        print('GPU:', torch.cuda.get_device_name(0))
        print('显存(GB):', round(torch.cuda.get_device_properties(0).total_memory / 1024 ** 3, 2))
except Exception as exc:
    print('Torch检查失败:', str(exc))
try:
    from packaging.version import Version
    tf = md.version('transformers')
    if Version(tf) < Version('4.51.0'):
        print('[注意] transformers低于4.51.0，不支持Qwen3；先不要盲目升级旧训练环境。')
except Exception:
    pass
print('磁盘剩余(GB):', round(shutil.disk_usage(ROOT).free / 1024**3, 2))
for sub in ('01_1p7b_non_thinking', '02_1p7b_thinking', '03_4b_non_thinking', '04_4b_thinking'):
    print('子项目', sub, '=>', 'OK' if (ROOT / sub).is_dir() else '缺失')
print('注意：此脚本仅做环境检查，不判断实际模型能否成功加载。')
