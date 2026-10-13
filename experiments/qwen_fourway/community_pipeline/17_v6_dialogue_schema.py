#!/usr/bin/env python3
"""QWbot V6 统一原始对话记录 schema。"""

import json
from pathlib import Path

REQUIRED_FIELDS = {"id", "source", "dialogue", "speaker_count", "topic_hint", "community_hint", "candidate_role"}
CANDIDATE_ROLES = {"finetune", "knowledge", "both", "unknown"}


def validate_record(row):
    if not isinstance(row, dict) or not REQUIRED_FIELDS.issubset(row):
        raise ValueError("对话记录缺少字段")
    if not isinstance(row["id"], str) or not row["id"].strip() or not isinstance(row["source"], str) or not row["source"].strip():
        raise ValueError("id/source必须是非空字符串")
    if not isinstance(row["dialogue"], list) or not row["dialogue"]:
        raise ValueError("dialogue必须是非空数组")
    if isinstance(row["speaker_count"], bool) or not isinstance(row["speaker_count"], int) or row["speaker_count"] < 1:
        raise ValueError("speaker_count错误")
    for field in ("topic_hint", "community_hint"):
        if row[field] is not None and not isinstance(row[field], str):
            raise ValueError("字段类型错误：" + field)
    if row["candidate_role"] not in CANDIDATE_ROLES:
        raise ValueError("candidate_role枚举错误")


def validate_records(rows):
    if not isinstance(rows, list):
        raise ValueError("记录集合必须是数组")
    ids = []
    for row in rows:
        validate_record(row)
        ids.append(row["id"])
    if len(ids) != len(set(ids)):
        raise ValueError("对话ID重复")


def read_jsonl(path):
    with Path(path).open(encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]
