#!/usr/bin/env python3
"""Offline, narrow hotfix for QWbot's existing public Bilibili pilot adapter.

Only changes 08_fetch_public_pilot.py, tests/test_public_pilot.py, and README.md.
Run from anywhere: python qwbot_public_pilot_hotfix.py /home/alex/Qwen_Finetune
Never contacts a network or touches raw corpus, older pipelines or blind tests.
"""
import argparse
import ast
from datetime import datetime
from pathlib import Path
import shutil
import sys
import tempfile

FETCH_ROWS = '''def fetch_rows(repo: str, limit: int, timeout: int = DEFAULT_TIMEOUT) -> tuple[list[dict], str, bytes]:
    """Fetch a bounded, nonrandom first-rows preview from a discovered config/split."""
    if repo not in SOURCES:
        raise ValueError(f"source is not allowlisted: {repo}")
    if not 1 <= limit <= 100:
        raise ValueError("limit must be between 1 and 100")

    splits_url = SPLITS_API + "?" + urllib.parse.urlencode({"dataset": repo})
    splits_response, _ = _json_request(splits_url, timeout)
    available = splits_response.get("splits")
    if not isinstance(available, list):
        raise RuntimeError(f"dataset splits not available: {splits_url}")
    choices = [entry for entry in available
               if isinstance(entry, dict)
               and isinstance(entry.get("config"), str) and entry["config"]
               and isinstance(entry.get("split"), str) and entry["split"]
               and entry.get("dataset", repo) == repo]
    if not choices:
        raise RuntimeError(f"no usable config/split found: {splits_url}")
    # Prefer train and default *when present*. Never assume either exists.
    chosen = sorted(choices, key=lambda e: (
        e["split"] != "train", e["config"] != "default", e["config"], e["split"]
    ))[0]
    config, split = chosen["config"], chosen["split"]
    query = urllib.parse.urlencode({"dataset": repo, "config": config, "split": split})
    url = ROWS_API + "?" + query
    data, raw = _json_request(url, timeout)
    rows = data.get("rows")
    if not isinstance(rows, list):
        raise RuntimeError(f"sample response has no rows list: {url}")
    out = []
    for idx, item in enumerate(rows[:limit]):
        if isinstance(item, dict) and isinstance(item.get("row"), dict):
            record = dict(item["row"])
            row_idx = item.get("row_idx")
            record["__qwb_preview_idx"] = row_idx if isinstance(row_idx, int) and row_idx >= 0 else idx
            record["__qwb_preview_config"] = config
            record["__qwb_preview_split"] = split
            out.append(record)
    return out, url, raw
'''

BASE_RECORD = '''def _base_record(repo: str, row: dict, index: int, source_file: str, declared_license: str) -> dict:
    original = row.get("message", "") if repo.startswith("Midsummra/") else row.get("output", "")
    safe_text, pflags = privacy(original if isinstance(original, str) else "")
    preview_idx = row.get("__qwb_preview_idx", index)
    if not isinstance(preview_idx, int) or preview_idx < 0:
        preview_idx = index
    config = row.get("__qwb_preview_config")
    split = row.get("__qwb_preview_split")
    record_key = f"{config}:{split}:{preview_idx}" if isinstance(config, str) and isinstance(split, str) else str(preview_idx)
    created = row.get("time", row.get("timestamp"))
    if isinstance(created, str):
        created = privacy(created)[0]
    # Missing conversation participants and reply edges remain explicitly unknown.
    return {
        "platform": "bilibili", "domain": [], "topics": [], "subtopics": [],
        "classification_confidence": 0.0, "classification_evidence": [], "classification_version": None,
        "thread_id": None, "message_id": f"public:{repo}:{record_key}",
        "parent_comment_id": None, "root_comment_id": None, "speaker_id": None,
        "message_order": None, "text": safe_text, "content_type": "comment",
        "created_at": created, "video_id": None, "post_id": None,
        "source_url": f"https://huggingface.co/datasets/{repo}", "source_file": str(Path(source_file).resolve()),
        "source_record_id": record_key, "provenance_type": "public_dataset_sample", "is_synthetic": False,
        "source_license": declared_license, "license_status": "dataset_declared_unverified" if declared_license != "unknown" else "verification_pending",
        "review_status": "unreviewed", "trainable": False,
        "quality_flags": ["no_confirmed_thread_id", "no_confirmed_parent_id", "no_confirmed_speaker_id", "first_rows_not_random"],
        "privacy_flags": pflags, "parse_status": "parsed_public_adapter", "split": None, "source_group_id": None,
        "public_dataset": repo,
    }
'''

ADAPT_ROWS = '''def adapt_rows(repo: str, rows: list[dict], source_file: str, declared_license: str = "unknown") -> list[dict]:
    if repo not in SOURCES:
        raise ValueError(f"source is not allowlisted: {repo}")
    records = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        if repo == "Midsummra/bilibilicomment":
            if not isinstance(row.get("message"), str) or not row["message"].strip():
                continue
        elif not isinstance(row.get("output"), str) or not row["output"].strip():
            continue
        record = _base_record(repo, row, index, source_file, declared_license)
        # Privacy flags from all exported text fields (not only record['text']).
        all_flags = set(record["privacy_flags"])
        def protected(value):
            if isinstance(value, str):
                safe, flags = privacy(value)
                all_flags.update(flags)
                return safe
            return value
        if repo == "Midsummra/bilibilicomment":
            record["source_fields"] = {k: protected(row.get(k)) for k in ("message", "time", "timestamp")}
            record["adapter_kind"] = "bilibili_independent_comment_candidate"
        else:
            for key in ("instruction", "input", "output"):
                record[key] = protected(row.get(key))
            record["relation_type"] = "derived_instruction_response"
            record["adapter_kind"] = "derived_instruction_response"
            record["quality_flags"].append("not_verified_multi_user_chat")
        record["privacy_flags"] = sorted(all_flags)
        records.append(record)
    return records
'''

NEW_TESTS = '''    def test_discover_real_config_and_preserve_preview_row_index(self):
        calls = []
        original = mod._json_request
        def fake(url, timeout):
            calls.append(url)
            if "/splits?" in url:
                return ({"splits": [{"dataset": "Midsummra/bilibilicomment", "config": "actual-config", "split": "validation"}]}, b"splits")
            return ({"rows": [{"row_idx": 42, "row": {"message": "炉石", "time": "2024"}}]}, b"rows")
        mod._json_request = fake
        try:
            rows, url, raw = mod.fetch_rows("Midsummra/bilibilicomment", 1)
            record = mod.adapt_rows("Midsummra/bilibilicomment", rows, "/tmp/example.jsonl")[0]
        finally:
            mod._json_request = original
        self.assertIn("config=actual-config", url)
        self.assertIn("split=validation", url)
        self.assertTrue(any("/splits?" in u for u in calls))
        self.assertIn("42", record["source_record_id"])
        self.assertFalse(record["trainable"])

    def test_nested_exported_text_fields_are_sanitized(self):
        phone = "13812345678"
        email = "privateperson@example.com"
        samples = [
            ("Midsummra/bilibilicomment", [{"message": "联系我" + phone, "time": "2024"}]),
            ("helloollel/chat_dataset_bilibili", [{"instruction": "邮箱" + email, "input": phone, "output": "回复" + email}]),
        ]
        for repo, rows in samples:
            result = mod.adapt_rows(repo, rows, "/tmp/example.jsonl")
            self.assertEqual(len(result), 1)
            serialized = json.dumps(result[0], ensure_ascii=False)
            self.assertNotIn(phone, serialized)
            self.assertNotIn(email, serialized)
            self.assertFalse(result[0]["trainable"])

'''


def replace_function(source: str, name: str, replacement: str) -> str:
    module = ast.parse(source)
    nodes = [n for n in module.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name]
    if len(nodes) != 1:
        raise RuntimeError(f"expected one function {name}, found {len(nodes)}")
    node = nodes[0]
    parts = source.splitlines(keepends=True)
    return "".join(parts[:node.lineno-1]) + replacement.rstrip() + "\n" + "".join(parts[node.end_lineno:])


def apply(root: Path) -> None:
    pipeline = root / "experiments/qwen_fourway/community_pipeline"
    fetch = pipeline / "08_fetch_public_pilot.py"
    tests = pipeline / "tests/test_public_pilot.py"
    readme = pipeline / "README.md"
    for path in (fetch, tests, readme):
        if not path.is_file():
            raise RuntimeError(f"required file missing: {path}")
    original = {p: p.read_text(encoding="utf-8") for p in (fetch, tests, readme)}

    text = original[fetch]
    if "SPLITS_API = " not in text:
        needle = 'ROWS_API = "https://datasets-server.huggingface.co/first-rows"'
        if needle not in text:
            raise RuntimeError("unrecognized ROWS_API declaration; no files changed")
        text = text.replace(needle, needle + '\nSPLITS_API = "https://datasets-server.huggingface.co/splits"', 1)
    for name, body in [("fetch_rows", FETCH_ROWS), ("_base_record", BASE_RECORD), ("adapt_rows", ADAPT_ROWS)]:
        text = replace_function(text, name, body)
    ast.parse(text)

    tests_text = original[tests]
    if "def test_discover_real_config_and_preserve_preview_row_index" not in tests_text:
        needle = '\n\nif __name__ == "__main__": unittest.main()'
        if needle not in tests_text:
            raise RuntimeError("unrecognized unit test footer; no files changed")
        tests_text = tests_text.replace(needle, '\n' + NEW_TESTS + needle, 1)
    ast.parse(tests_text)

    readme_text = original[readme]
    old = '''OUT="$P/output_public_pilot_$(date +%Y%m%d_%H%M%S)"

# 1) 仅检查两个来源的元数据（不会获取样本）
.venv/bin/python "$P/08_fetch_public_pilot.py" --source all --output-dir "$OUT"

# 2) 获取每个来源最多 100 行（不会下载完整数据集）
.venv/bin/python "$P/08_fetch_public_pilot.py" --source all --download-pilot --limit 100 --output-dir "$OUT"'''
    new = '''META_OUT="$P/output_public_metadata_$(date +%Y%m%d_%H%M%S)"
OUT="$P/output_public_pilot_$(date +%Y%m%d_%H%M%S)"

# 1) 仅检查两个来源的元数据（不会获取样本）
.venv/bin/python "$P/08_fetch_public_pilot.py" --source all --output-dir "$META_OUT"

# 2) 使用全新目录获取每个来源最多100条头部预览（非随机样本）
#    --download-pilot本身会再次保存元数据，因此不能复用META_OUT。
.venv/bin/python "$P/08_fetch_public_pilot.py" --source all --download-pilot --limit 100 --output-dir "$OUT"'''
    if old not in readme_text:
        if new not in readme_text:
            raise RuntimeError("unrecognized README command sequence; no files changed")
    else:
        readme_text = readme_text.replace(old, new, 1)
    if "first-rows为数据集前100条预览" not in readme_text:
        readme_text += "\n\n注意：first-rows为数据集前100条预览，非随机抽样；config/split通过/splits查询，不预设default/train。\n"

    changes = {fetch: text, tests: tests_text, readme: readme_text}
    backup = Path(tempfile.mkdtemp(prefix="qwbot_public_pilot_hotfix_backup_"))
    for path in changes:
        shutil.copy2(path, backup / path.name)
    try:
        for path, contents in changes.items():
            # Write temp then atomically replace, preserving old version in /tmp.
            tmp = path.with_name(path.name + ".hotfix.tmp")
            try:
                tmp.write_text(contents, encoding="utf-8")
                tmp.replace(path)
            finally:
                tmp.unlink(missing_ok=True)
    except Exception:
        for path in original:
            shutil.copy2(backup / path.name, path)
        raise
    print("PATCH_APPLIED: 08_fetch_public_pilot.py, tests/test_public_pilot.py, README.md")
    print("BACKUP_DIR:", backup)
    print("NO_NETWORK_OR_DOWNLOAD_PERFORMED")
    print("Next run: python -m unittest discover -s experiments/qwen_fourway/community_pipeline/tests -v")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("project_root", type=Path)
    args = parser.parse_args()
    try:
        apply(args.project_root.expanduser().resolve())
    except Exception as exc:
        sys.exit(f"PATCH_FAILED: {exc}")
