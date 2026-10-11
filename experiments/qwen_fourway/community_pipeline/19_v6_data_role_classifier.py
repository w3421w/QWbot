#!/usr/bin/env python3
"""QWbot V6 统一对话数据用途分类器（纯本地启发式，不联网）。"""
from __future__ import annotations

import re
from typing import Any

DATA_ROLES = {"finetune", "knowledge", "discard", "review"}

_MEME_WORDS = ("梗", "meme", "出处", "什么梗", "怎么来的", "含义", "典故")
_EXPLANATION_WORDS = ("指的是", "意思是", "解释", "来源", "原意", "指代")
_NOISE_RE = re.compile(r"^(哈+|呵+|额+|嗯+|啊+|哦+|。。。+|\.+|[!！?？~～\s]+)$", re.I)


def _texts(record: dict[str, Any]) -> list[str]:
    dialogue = record.get("dialogue", [])
    if not isinstance(dialogue, list):
        return []
    return [x.get("text", "").strip() for x in dialogue
            if isinstance(x, dict) and isinstance(x.get("text"), str)]


def _tags(record: dict[str, Any], texts: list[str]) -> list[str]:
    tags: list[str] = []
    topic = record.get("topic_hint")
    source = str(record.get("source", ""))
    if any(any(word.lower() in text.lower() for word in _MEME_WORDS) for text in texts) or "meme" in source.lower():
        tags.append("meme_explanation")
    if len({x.get("speaker") for x in record.get("dialogue", []) if isinstance(x, dict)}) >= 2:
        tags.append("multi_turn")
    if sum(len(x) for x in texts) >= 30:
        tags.append("substantive")
    if isinstance(topic, str) and topic.strip():
        tags.append("topic_annotated")
    return tags


def classify_record(record: dict[str, Any]) -> dict[str, Any]:
    """返回用途分类；输入的 candidate_role 原值不会被修改。"""
    if not isinstance(record, dict):
        raise ValueError("record必须是对象")
    texts = _texts(record)
    tags = _tags(record, texts)
    community = record.get("community_hint")
    community_hint = community if isinstance(community, str) and community.strip() else None
    nonempty = [t for t in texts if t]
    total = sum(len(t) for t in nonempty)

    if not nonempty:
        role, confidence, reason = "discard", 0.99, "没有可用文本"
    elif all(_NOISE_RE.match(t) or len(t) <= 1 for t in nonempty) or total < 4:
        role, confidence, reason = "discard", 0.96, "文本过短或仅包含无意义噪声"
    elif ("meme_explanation" in tags and
          (any(w in " ".join(nonempty).lower() for w in _EXPLANATION_WORDS) or
           len(nonempty) == 1)):
        role, confidence, reason = "knowledge", 0.91, "内容呈现梗及其含义、出处或解释"
    elif len({x.get("speaker") for x in record.get("dialogue", []) if isinstance(x, dict)}) >= 2 and len(nonempty) >= 2 and total >= 12:
        role, confidence, reason = "finetune", 0.86, "多人多轮且文本具有足够语义内容"
    else:
        role, confidence, reason = "review", 0.55, "无法从文本特征可靠确定最终用途"

    return {"id": record.get("id"), "data_role": role, "confidence": confidence,
            "style_tags": tags, "community_hint": community_hint,
            "quality_reason": reason}


classify = classify_record


if __name__ == "__main__":
    raise SystemExit("请在代码中调用 classify_record(record)")
