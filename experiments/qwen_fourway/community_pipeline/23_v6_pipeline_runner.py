#!/usr/bin/env python3
"""QWbot V6 一键本地数据流水线入口。"""
from __future__ import annotations
import argparse, importlib.util, json, shutil, subprocess, sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent

def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    mod = importlib.util.module_from_spec(spec); assert spec.loader
    spec.loader.exec_module(mod); return mod


def _count(path: Path) -> int:
    try:
        
        with path.open(encoding='utf-8') as handle:
            return sum(1 for line in handle if line.strip())
    except OSError:
        return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('input', type=Path)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--keep-intermediate', action='store_true', default=True)
    p.add_argument('--no-keep-intermediate', dest='keep_intermediate', action='store_false')
    args = p.parse_args(argv)
    root = args.output_dir.resolve()
    for d in ('dialogue','errors','reports','classified','sft','split'): (root/d).mkdir(parents=True, exist_ok=True)
    errors: list[dict[str, Any]] = []
    status: dict[str, str] = {}
    raw = root/'dialogue/v6_dialogue_raw.jsonl'; classified = root/'classified/v6_classified.jsonl'; sft = root/'sft/qwen_sft_train.jsonl'
    try:
        m = _load('v6_adapter', '18_v6_batch_adapter.py')
        m.convert_jsonl(args.input, raw, root/'errors/adapter_errors.jsonl', root/'reports/adapter_report.json')
        status['adapter'] = 'success'
    except Exception as exc:
        status['adapter'] = 'failed'; errors.append({'stage':'adapter','error':str(exc)})
    try:
        m = _load('v6_classifier', '19_v6_data_role_classifier.py')
        with raw.open(encoding='utf-8') as src, classified.open('w', encoding='utf-8') as dst:
            for line in src:
                if line.strip():
                    row = json.loads(line); out = m.classify_record(row); out['id'] = row.get('id'); dst.write(json.dumps(out, ensure_ascii=False)+'\n')
        status['classifier'] = 'success'
    except Exception as exc:
        status['classifier'] = 'failed'; errors.append({'stage':'classifier','error':str(exc)})
    try:
        m = _load('v6_quality', '20_v6_quality_report.py')
        m.generate_quality_report(raw, classified, root/'reports/v6_quality_report.json')
        status['quality_report'] = 'success'
    except Exception as exc:
        status['quality_report'] = 'failed'; errors.append({'stage':'quality_report','error':str(exc)})
    try:
        m = _load('v6_export', '21_v6_export_sft.py')
        m.export_sft(raw, classified, sft, root/'sft/export_errors.jsonl', root/'sft/export_report.json')
        status['export_sft'] = 'success'
    except Exception as exc:
        status['export_sft'] = 'failed'; errors.append({'stage':'export_sft','error':str(exc)})
    try:
        m = _load('v6_split', '22_v6_dataset_split.py')
        m.split_dataset(sft, root/'split', seed=args.seed)
        split_report = root/'split/split_report.json'
        if split_report.exists():
            data = json.loads(split_report.read_text(encoding='utf-8')); data['seed'] = args.seed
            split_report.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        status['dataset_split'] = 'success'
    except Exception as exc:
        status['dataset_split'] = 'failed'; errors.append({'stage':'dataset_split','error':str(exc)})
    (root/'pipeline_errors.json').write_text(json.dumps(errors, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    report = {'stage_status': status, 'input_count': _count(args.input), 'output_count': _count(raw), 'final_train_count': _count(root/'split/train.jsonl'), 'final_valid_count': _count(root/'split/validation.jsonl'), 'final_test_count': _count(root/'split/test.jsonl')}
    (root/'pipeline_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    if not args.keep_intermediate:
        for path in (raw, classified):
            try: path.unlink()
            except FileNotFoundError: pass
    return 0 if not errors else 1

if __name__ == '__main__': raise SystemExit(main())
