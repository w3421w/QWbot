#!/usr/bin/env python3
"""Re-mask the existing public pilot without modifying its original files."""
import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from common.core import privacy

P = Path(__file__).resolve().parent

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(65536), b''):
            h.update(block)
    return h.hexdigest()

def sanitize(value, flags):
    if isinstance(value, str):
        safe, new_flags = privacy(value)
        flags.update(new_flags)
        return safe
    if isinstance(value, list):
        return [sanitize(v, flags) for v in value]
    if isinstance(value, dict):
        return {k: sanitize(v, flags) for k, v in value.items()}
    return value

def write_jsonl_new(path, records):
    with path.open('x', encoding='utf-8') as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    os.chmod(path, 0o600)

def run(original, dest):
    original = Path(original).resolve()
    dest = Path(dest).resolve()

    inputs = sorted(original.glob('*.pilot.jsonl'))
    if not original.is_dir() or len(inputs) != 2:
        raise RuntimeError(
            'Expected exactly two original pilot JSONL files'
        )
    if dest.exists():
        raise FileExistsError(f'Will not overwrite: {dest}')

    data = []
    for source in inputs:
        rows = []
        with source.open(encoding='utf-8-sig') as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    if not isinstance(record, dict):
                        raise ValueError('Invalid record: ' + str(source))
                    rows.append(record)
        data.append((source, rows))

    spec = importlib.util.spec_from_file_location(
        'qwbot_review_v2', P / '07_export_review.py'
    )
    exporter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exporter)

    dest.mkdir(mode=0o700, parents=True, exist_ok=False)
    os.chmod(dest, 0o700)

    report = {
        'input_dir': str(original),
        'trainable': False,
        'policy': 'privacy-v2; ambiguous mentions quarantined',
        'sources': []
    }

    for src, rows in data:
        processed = []
        quarantine = []

        for old in rows:
            r = dict(old)
            flags = set(r.get('privacy_flags') or [])

            for key in (
                'text', 'instruction', 'input',
                'output', 'source_fields', 'created_at'
            ):
                if key in r:
                    r[key] = sanitize(r[key], flags)

            r['privacy_flags'] = sorted(flags)
            r['trainable'] = False

            if (
                'ambiguous_mention' in flags
                or 'private_identity' in flags
            ):
                r['review_status'] = 'privacy_quarantine'
                quality = list(r.get('quality_flags') or [])

                if 'privacy_quarantine' not in quality:
                    quality.append('privacy_quarantine')

                r['quality_flags'] = quality
                quarantine.append(r)

            processed.append(r)

        clean_file = dest / src.name
        quarantine_file = dest / src.name.replace(
            '.pilot.jsonl', '.quarantine.jsonl'
        )
        csv_file = dest / src.name.replace(
            '.pilot.jsonl', '.review.csv'
        )

        write_jsonl_new(clean_file, processed)
        write_jsonl_new(quarantine_file, quarantine)

        exporter.export(processed, csv_file)
        os.chmod(csv_file, 0o600)

        with csv_file.open(
            encoding='utf-8-sig', newline=''
        ) as f:
            review_count = sum(1 for _ in csv.DictReader(f))

        report['sources'].append({
            'input': src.name,
            'total': len(rows),
            'quarantined': len(quarantine),
            'review_count': review_count,
            'input_sha256': sha256(src),
            'output_sha256': sha256(clean_file)
        })

    manifest = dest / 'privacy_v2_manifest.json'
    with manifest.open('x', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write('\n')
    os.chmod(manifest, 0o600)

    print(json.dumps(report, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--source-dir', type=Path, required=True)
    ap.add_argument('--output-dir', type=Path, required=True)
    args = ap.parse_args()
    run(args.source_dir, args.output_dir)
