#!/usr/bin/env python3
"""QWbot V6 原始数据来源适配器。

本模块只做内存中的格式转换，不进行网络访问、下载或隐私处理。
"""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
from typing import Any

_SCHEMA_PATH = Path(__file__).with_name("17_v6_dialogue_schema.py")
_spec = importlib.util.spec_from_file_location("v6_dialogue_schema", _SCHEMA_PATH)
_schema = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_schema)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _stable_id(source: str, value: Any) -> str:
    raw = str(value)
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
    return f"{source}:{digest}"


def _record(record_id: str, source: str, dialogue: list[dict[str, str]], *,
            community_hint: str | None, candidate_role: str,
            topic_hint: str | None = None) -> dict[str, Any]:
    _require(dialogue and all(isinstance(x, dict) and isinstance(x.get("speaker"), str)
             and isinstance(x.get("text"), str) and x["speaker"].strip() and x["text"].strip()
             for x in dialogue), "dialogue必须是非空的speaker/text数组")
    row = {"id": record_id, "source": source, "dialogue": dialogue,
           "speaker_count": len({x["speaker"] for x in dialogue}),
           "topic_hint": topic_hint, "community_hint": community_hint,
           "candidate_role": candidate_role}
    _schema.validate_record(row)
    return row


def adapt_bilibili_reply_chain(payload: dict[str, Any]) -> dict[str, Any]:
    """将 {oid, comments:[{user,text}]} 转为统一记录。"""
    _require(isinstance(payload, dict), "bilibili_reply_chain输入必须是对象")
    oid = payload.get("oid")
    comments = payload.get("comments")
    _require(isinstance(oid, (str, int)) and str(oid).strip(), "oid必须存在")
    _require(isinstance(comments, list) and comments, "comments必须是非空数组")
    dialogue = []
    for item in comments:
        _require(isinstance(item, dict), "评论必须是对象")
        user, text = item.get("user"), item.get("text")
        _require(isinstance(user, str) and user.strip() and isinstance(text, str) and text.strip(),
                 "评论必须包含非空user和text")
        dialogue.append({"speaker": user.strip(), "text": text.strip()})
    return _record(f"bilibili_reply_chain:{oid}", "bilibili_reply_chain", dialogue,
                   community_hint="bilibili", candidate_role="both")


def adapt_lccc_chat(payload: list[str] | dict[str, Any]) -> dict[str, Any]:
    """将多轮字符串数组转为交替 speaker 的统一记录。"""
    if isinstance(payload, dict):
        turns = payload.get("dialogue", payload.get("messages"))
    else:
        turns = payload
    _require(isinstance(turns, list) and turns, "lccc_chat必须是非空字符串数组")
    _require(all(isinstance(x, str) and x.strip() for x in turns), "聊天轮次必须是非空字符串")
    dialogue = [{"speaker": f"speaker_{i % 2 + 1}", "text": text.strip()}
                for i, text in enumerate(turns)]
    return _record(_stable_id("lccc_chat", "\n".join(turns)), "lccc_chat", dialogue,
                   community_hint="internet_general", candidate_role="finetune")


def adapt_knowledge_only(payload: Any) -> dict[str, Any]:
    """将 MemeBench/CHIME 等知识样本转换为知识占位记录。"""
    _require(isinstance(payload, dict), "knowledge_only输入必须是对象")
    source = payload.get("source", "knowledge_only")
    _require(isinstance(source, str) and source.strip(), "source必须是非空字符串")
    value = payload.get("id", payload.get("uid", payload.get("text", payload.get("content"))))
    _require(value is not None and str(value).strip(), "知识样本必须包含id或内容")
    text = payload.get("text", payload.get("content", payload.get("knowledge", "")))
    if not isinstance(text, str) or not text.strip():
        text = "knowledge placeholder"
    return _record(_stable_id(source, payload.get("id", value)), source,
                   [{"speaker": "knowledge", "text": text.strip()}],
                   community_hint="internet_general", candidate_role="knowledge",
                   topic_hint=payload.get("topic_hint") if isinstance(payload.get("topic_hint"), str) else None)


def adapt_source(source_type: str, payload: Any) -> dict[str, Any]:
    _require(isinstance(source_type, str), "source_type必须是字符串")
    adapters = {"bilibili_reply_chain": adapt_bilibili_reply_chain,
                "lccc_chat": adapt_lccc_chat, "knowledge_only": adapt_knowledge_only}
    _require(source_type in adapters, f"不支持的来源类型: {source_type}")
    return adapters[source_type](payload)


# 便于调用者使用简短统一名称
adapt = adapt_source
