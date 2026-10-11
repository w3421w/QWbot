#!/usr/bin/env python3
"""QWbot V6 公开中文互联网数据源登记表（只记录元数据，不下载数据）。"""

import argparse
import json
import os
from pathlib import Path

P = Path(__file__).resolve().parent
REGISTRY_PATH = P / "v6_source_registry.json"

SOURCE_TYPES = {"community_comments", "dialogue_dataset", "benchmark_dataset"}
CANDIDATE_ROLES = {"finetune", "knowledge", "both", "unknown"}
LICENSE_STATUSES = {"verified", "research_only", "verification_pending", "unknown"}
QUALITY_RATINGS = {1, 2, 3, 4, 5}
REQUIRED_FIELDS = {"source_id", "name", "url", "source_type", "candidate_role",
                   "community", "license_status", "quality_rating", "notes"}

DEFAULT_SOURCES = [
    {"source_id": "bilibili_comment_reply_chains", "name": "Bilibili评论回复链",
     "url": "https://www.bilibili.com/", "source_type": "community_comments",
     "candidate_role": "both", "community": "bilibili", "license_status": "verification_pending",
     "quality_rating": 4, "notes": "评论与回复上下文适合对话候选；需确认公开数据许可、隐私与抓取合规。"},
    {"source_id": "bilibili_large_scale_comments", "name": "Bilibili大规模评论数据",
     "url": "https://www.bilibili.com/", "source_type": "community_comments",
     "candidate_role": "both", "community": "bilibili", "license_status": "verification_pending",
     "quality_rating": 3, "notes": "规模大但噪声和重复较多，需先做去重、隐私和质量过滤。"},
    {"source_id": "lccc_chinese_chat", "name": "LCCC中文聊天数据",
     "url": "https://github.com/thu-coai/LCCC", "source_type": "dialogue_dataset",
     "candidate_role": "finetune", "community": "internet_general", "license_status": "verification_pending",
     "quality_rating": 4, "notes": "中文多轮聊天数据；使用前核对原始许可和再分发条件。"},
    {"source_id": "memebench", "name": "MemeBench",
     "url": "https://github.com/haotiansun14/MemeBench", "source_type": "benchmark_dataset",
     "candidate_role": "knowledge", "community": "internet_general", "license_status": "verification_pending",
     "quality_rating": 4, "notes": "梗知识评测来源，默认作为知识候选或评测保留，不直接用于对话微调。"},
    {"source_id": "chime", "name": "CHIME",
     "url": "https://github.com/thu-coai/CHIME", "source_type": "dialogue_dataset",
     "candidate_role": "knowledge", "community": "internet_general", "license_status": "verification_pending",
     "quality_rating": 3, "notes": "中文互联网梗/知识相关数据；需核对字段、许可和隐私后再分流。"},
]


def validate_source(row):
    if not isinstance(row, dict) or not REQUIRED_FIELDS.issubset(row):
        raise ValueError("来源记录缺少字段")
    for field in ("source_id", "name", "url", "source_type", "candidate_role", "community", "license_status", "notes"):
        if not isinstance(row[field], str) or not row[field].strip():
            raise ValueError("来源字段类型或内容错误：" + field)
    if row["source_type"] not in SOURCE_TYPES or row["candidate_role"] not in CANDIDATE_ROLES:
        raise ValueError("来源枚举错误")
    if row["license_status"] not in LICENSE_STATUSES:
        raise ValueError("许可状态枚举错误")
    if isinstance(row["quality_rating"], bool) or row["quality_rating"] not in QUALITY_RATINGS:
        raise ValueError("质量评级错误")


def validate_registry(rows):
    if not isinstance(rows, list) or not rows:
        raise ValueError("来源登记表不能为空")
    ids = []
    for row in rows:
        validate_source(row)
        ids.append(row["source_id"])
    if len(ids) != len(set(ids)):
        raise ValueError("source_id重复")


def write_registry(path=REGISTRY_PATH, rows=None):
    rows = DEFAULT_SOURCES if rows is None else rows
    validate_registry(rows)
    path = Path(path)
    if path.exists():
        raise FileExistsError("拒绝覆盖：" + str(path))
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.chmod(path, 0o600)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(REGISTRY_PATH))
    args = parser.parse_args()
    path = write_registry(args.output)
    print(json.dumps({"output": str(path), "count": len(DEFAULT_SOURCES)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
