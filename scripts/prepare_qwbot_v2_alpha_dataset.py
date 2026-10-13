#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prepare the first QWbot V2 Alpha Qwen3/Qwen4B SFT dataset.

This script is intentionally an export-only step.  It reads the expansion SFT
file, applies conservative quality checks, performs a group-aware train/validation
split, and writes the three Alpha deliverables.  It never starts training and
does not import or modify any older preparation script.

The expansion JSONL itself contains only ``messages``.  Its metadata sidecar is
therefore joined by record order so that ``source_id`` and ``category`` can be
used for leakage-safe splitting.  A ``-c1``/``-c2``/``-c3`` suffix denotes
rewrites of one source context; those rows are put in one group.  Exact context
fingerprints are also merged across group keys as a final leakage guard.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import random
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "datasets"
DEFAULT_INPUT = DATA_DIR / "qwbot_v2_expansion_sft_v1.jsonl"
DEFAULT_METADATA = DATA_DIR / "qwbot_v2_expansion_metadata.jsonl"
DEFAULT_TRAIN = DATA_DIR / "qwbot_v2_alpha_train.jsonl"
DEFAULT_VALIDATION = DATA_DIR / "qwbot_v2_alpha_val.jsonl"
DEFAULT_REPORT = DATA_DIR / "qwbot_v2_alpha_report.json"

DEFAULT_SEED = 20261013
DEFAULT_VAL_RATIO = 0.10
PREVIEW_SIZE = 50

# Keep the established QWbot persona as one stable system message in every
# exported row.  The source file currently uses this exact persona as well.
SYSTEM_PERSONA = """你是QQ群里的普通群友QWbot。

要求：

- 像真实群友聊天；
- 先理解上下文；
- 判断是否需要回复；
- 不抢普通闲聊；
- 回复简短自然；
- 可以接梗、吐槽、轻度阴阳；
- 不强行使用黑话；
- 不像客服；
- 不输出长篇解释；
- 用户认真求助时正常回应；
- 不编造事实。"""


FILTER_REASON_KEYS = (
    "messages结构错误",
    "system缺失",
    "user为空",
    "assistant为空",
    "user/assistant高度重复",
    "重复pair",
    "AI腔",
    "客服腔",
    "JSON无效",
)

# These patterns are deliberately narrow.  A casual mention of a model or a
# database can be a normal group joke; the patterns below target explanatory
# assistant voice and customer-support boilerplate instead.
AI_TONE_RE = re.compile(
    r"作为(?:一个)?(?:AI|人工智能|语言模型)|作为模型|语言模型|"
    r"(?:我是|本来也是|我这边是)(?:一个)?(?:AI|人工智能|语言模型)|"
    r"根据(?:你的问题|你的描述|以上信息|您提供的)|综上所述|"
    r"以下(?:几点|是.{0,20})|我(?:的)?数据库|搜索API|Tavily|"
    r"希望.{0,20}对你有帮助|"
    r"(?:无法|不能|没法)(?:联网|访问|提供).{0,12}(?:资料|数据|信息)",
    re.IGNORECASE | re.DOTALL,
)
CUSTOMER_SERVICE_RE = re.compile(
    r"您好|尊敬的(?:用户|客户)|感谢您的(?:反馈|咨询|理解)|"
    r"很高兴为您(?:服务|解答)|为您(?:服务|解答|查询|处理)|"
    r"请稍候|稍等片刻|已为您(?:查询|处理)|温馨提示|"
    r"如有(?:任何|其他)?问题.*(?:联系|咨询)|提交工单|祝您生活愉快",
    re.IGNORECASE | re.DOTALL,
)


@dataclass
class Candidate:
    """A validated source row plus metadata used only during preparation."""

    input_index: int
    line_number: int
    user: str
    assistant: str
    source_id: str
    category: str
    source: str
    metadata: dict[str, Any]
    group_key: str = ""
    context_key: str = ""
    split: str = ""

    def messages(self) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": SYSTEM_PERSONA},
            {"role": "user", "content": self.user},
            {"role": "assistant", "content": self.assistant},
        ]

    def output_row(self) -> dict[str, Any]:
        return {"messages": self.messages()}


def clean_text(value: Any) -> str:
    """Normalize text without destroying QQ speaker/context line breaks."""

    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value))
    text = text.replace("\ufeff", "").replace("\u200b", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    return "\n".join(line for line in lines if line).strip()


def norm_key(value: Any) -> str:
    """A whitespace/punctuation-insensitive key for duplicate checks."""

    text = clean_text(value).casefold()
    return re.sub(r"\s+", "", text)


def content_key(value: Any) -> str:
    """A stronger key for exact context leakage checks."""

    text = norm_key(value)
    return re.sub(r"[^\w\u3400-\u9fff]", "", text, flags=re.UNICODE)


def stable_hash(value: str) -> int:
    return int(hashlib.sha1(value.encode("utf-8")).hexdigest()[:16], 16)


def root_source_id(source_id: str) -> str:
    """Collapse rewrite suffixes while retaining source-specific identities."""

    source_id = clean_text(source_id)
    return re.sub(r"(?:-c\d+)+$", "", source_id, flags=re.IGNORECASE) or "unknown"


def read_jsonl_rows(path: Path) -> tuple[list[tuple[int, dict[str, Any]]], int, int]:
    """Return valid dict rows, physical non-empty count, and invalid count."""

    rows: list[tuple[int, dict[str, Any]]] = []
    nonempty = 0
    invalid = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            nonempty += 1
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                invalid += 1
                continue
            if not isinstance(value, dict):
                invalid += 1
                continue
            rows.append((line_number, value))
    return rows, nonempty, invalid


def read_metadata(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.is_file():
        return []
    metadata: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                metadata.append(value)
    return metadata


def latest_human_message(user: str) -> str:
    """Extract the last [群友X] line for repetition checks.

    The complete user context contains previous QWbot replies, so comparing an
    assistant answer against the whole context would incorrectly flag normal
    follow-up answers.  Only the latest human utterance is used here.
    """

    lines = user.splitlines()
    speaker_re = re.compile(r"^\s*\[([^\]]+)\]\s*(.*)$", re.I)
    current_kind = ""
    current_parts: list[str] = []
    latest = ""
    for line in lines:
        match = speaker_re.match(line)
        if match:
            label = clean_text(match.group(1)).casefold()
            is_human = "群友" in label or "用户" in label or label.startswith("user")
            if is_human:
                current_kind = "human"
                current_parts = [clean_text(match.group(2))] if match.group(2).strip() else []
                latest = clean_text("\n".join(current_parts))
            else:
                current_kind = "other"
                current_parts = []
            continue
        if current_kind == "human":
            current_parts.append(clean_text(line))
            latest = clean_text("\n".join(current_parts))
    if latest:
        return latest
    return clean_text(lines[-1] if lines else "")


def compact_for_similarity(value: str) -> str:
    return re.sub(r"[^0-9A-Za-z\u3400-\u9fff]", "", clean_text(value)).casefold()


def highly_repetitive(user: str, assistant: str) -> bool:
    """Detect an answer that mostly copies the latest user utterance.

    Short replies are intentionally exempt: a natural ``收到`` or ``不是`` is
    useful training data even when it overlaps with the prompt.
    """

    latest = compact_for_similarity(latest_human_message(user))
    answer = compact_for_similarity(assistant)
    if len(latest) < 8 or len(answer) < 8:
        return False
    ratio = difflib.SequenceMatcher(None, latest, answer).ratio()
    if ratio >= 0.92:
        return True
    if len(answer) >= 10 and answer in latest:
        return True
    if len(latest) >= 10 and latest in answer:
        return True
    return False


def parse_messages(value: Any) -> tuple[str, str, str | None]:
    """Return (user, assistant, structural_error).

    Alpha rows intentionally have exactly system/user/assistant in that order;
    accepting extra turns would silently violate the requested final format.
    """

    if not isinstance(value, list) or len(value) != 3:
        return "", "", "messages结构错误"
    if not all(isinstance(item, dict) for item in value):
        return "", "", "messages结构错误"
    roles = [item.get("role") for item in value]
    if roles != ["system", "user", "assistant"]:
        return "", "", "messages结构错误"
    if any(not isinstance(item.get("content"), str) for item in value):
        return "", "", "messages结构错误"
    system = clean_text(value[0]["content"])
    user = clean_text(value[1]["content"])
    assistant = clean_text(value[2]["content"])
    if not system:
        return user, assistant, "system缺失"
    return user, assistant, None


def metadata_for(
    row: dict[str, Any], metadata: dict[str, Any] | None, input_index: int
) -> tuple[str, str, str, dict[str, Any], bool]:
    """Resolve source/category fields from row metadata, with safe fallbacks."""

    metadata = metadata or {}
    # A future input export may carry metadata inline; inline fields take
    # precedence only when the sidecar does not provide a non-empty value.
    def first_nonempty(*values: Any) -> str:
        for value in values:
            text = clean_text(value)
            if text:
                return text
        return ""

    source_id = first_nonempty(
        metadata.get("source_id"), row.get("source_id"), row.get("id"), f"line-{input_index + 1}"
    )
    category = first_nonempty(metadata.get("category"), row.get("category"), "未分类")
    source = first_nonempty(metadata.get("source"), row.get("source"), "input")
    missing = not bool(metadata)
    return source_id, category, source, metadata, missing


def quality_reason(user: str, assistant: str) -> str | None:
    """Return one deterministic filter reason, or None when usable."""

    if not user:
        return "user为空"
    if not assistant:
        return "assistant为空"
    if highly_repetitive(user, assistant):
        return "user/assistant高度重复"
    if AI_TONE_RE.search(assistant):
        return "AI腔"
    if CUSTOMER_SERVICE_RE.search(assistant):
        return "客服腔"
    return None


def build_group_components(candidates: list[Candidate]) -> dict[int, list[int]]:
    """Union source/category groups and identical contexts into components."""

    parent = list(range(len(candidates)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    first_by_key: dict[str, int] = {}
    for index, candidate in enumerate(candidates):
        root = root_source_id(candidate.source_id)
        candidate.group_key = f"{root}\0{candidate.category}"
        candidate.context_key = content_key(candidate.user)
        for key in (candidate.group_key, f"context\0{candidate.context_key}"):
            if key and key != "context\0":
                previous = first_by_key.get(key)
                if previous is None:
                    first_by_key[key] = index
                else:
                    union(index, previous)

    components: dict[int, list[int]] = defaultdict(list)
    for index in range(len(candidates)):
        components[find(index)].append(index)
    return dict(components)


def choose_validation_groups(
    candidates: list[Candidate], components: dict[int, list[int]], val_ratio: float, seed: int
) -> set[int]:
    """Choose whole components near the requested validation size.

    Components are shuffled by a seeded RNG and added only while they fit the
    target.  A small nearest-size repair pass handles the uncommon case where a
    large component would otherwise leave an avoidable gap.
    """

    total = len(candidates)
    if total <= 1 or val_ratio <= 0:
        return set()
    target = max(1, min(total - 1, int(round(total * val_ratio))))
    keys = list(components)
    rng = random.Random(seed)
    rng.shuffle(keys)
    selected: set[int] = set()
    selected_count = 0
    for key in keys:
        size = len(components[key])
        if selected_count + size <= target:
            selected.add(key)
            selected_count += size
        if selected_count == target:
            return selected

    # If the greedy pass missed the exact target, add the one component that
    # gets closest, even if it crosses the target.  All other components stay
    # in train, so no data is discarded.
    remaining = [key for key in keys if key not in selected]
    if remaining and selected_count < target:
        best = min(remaining, key=lambda key: abs(target - (selected_count + len(components[key]))))
        selected.add(best)
    return selected


def split_candidates(candidates: list[Candidate], val_ratio: float, seed: int) -> tuple[int, int, int]:
    components = build_group_components(candidates)
    validation_roots = choose_validation_groups(candidates, components, val_ratio, seed)
    for root, indexes in components.items():
        split = "validation" if root in validation_roots else "train"
        for index in indexes:
            candidates[index].split = split
    train_count = sum(candidate.split == "train" for candidate in candidates)
    val_count = sum(candidate.split == "validation" for candidate in candidates)
    return train_count, val_count, len(components)


def distribution(candidates: Iterable[Candidate], field: str, split: str | None = None) -> dict[str, int]:
    values: Counter[str] = Counter()
    for candidate in candidates:
        if split is not None and candidate.split != split:
            continue
        values[str(getattr(candidate, field))] += 1
    return dict(sorted(values.items(), key=lambda item: (-item[1], item[0])))


def preview_item(candidate: Candidate) -> dict[str, Any]:
    return {
        "input_index": candidate.input_index,
        "source_id": candidate.source_id,
        "category": candidate.category,
        "source": candidate.source,
        "split": candidate.split,
        "messages": candidate.messages(),
    }


def length_item(candidate: Candidate) -> dict[str, Any]:
    return {
        "input_index": candidate.input_index,
        "source_id": candidate.source_id,
        "category": candidate.category,
        "assistant_length": len(candidate.assistant),
        "assistant": candidate.assistant,
    }


def write_jsonl(path: Path, candidates: Iterable[Candidate]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for candidate in candidates:
            handle.write(json.dumps(candidate.output_row(), ensure_ascii=False, separators=(",", ":")) + "\n")


def prepare(
    input_path: Path,
    metadata_path: Path | None,
    train_path: Path,
    validation_path: Path,
    report_path: Path,
    val_ratio: float = DEFAULT_VAL_RATIO,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    if not 0 <= val_ratio < 1:
        raise ValueError("val_ratio must be in [0, 1)")
    if not input_path.is_file():
        raise FileNotFoundError(f"input not found: {input_path}")

    raw_rows, total_input, invalid_rows = read_jsonl_rows(input_path)
    metadata_rows = read_metadata(metadata_path if metadata_path and metadata_path.is_file() else None)
    candidates: list[Candidate] = []
    filter_reasons: Counter[str] = Counter()
    filtered_rows = 0
    metadata_missing_rows = 0
    duplicate_pair_count = 0
    seen_pairs: set[tuple[str, str]] = set()

    for ordinal, (line_number, row) in enumerate(raw_rows):
        user, assistant, structure_reason = parse_messages(row.get("messages"))
        if structure_reason:
            filter_reasons[structure_reason] += 1
            filtered_rows += 1
            continue

        sidecar = metadata_rows[ordinal] if ordinal < len(metadata_rows) else None
        source_id, category, source, metadata, missing_metadata = metadata_for(row, sidecar, ordinal)
        if missing_metadata:
            metadata_missing_rows += 1

        reason = quality_reason(user, assistant)
        if reason:
            filter_reasons[reason] += 1
            filtered_rows += 1
            continue

        pair = (norm_key(user), norm_key(assistant))
        if pair in seen_pairs:
            filter_reasons["重复pair"] += 1
            duplicate_pair_count += 1
            filtered_rows += 1
            continue
        seen_pairs.add(pair)

        candidates.append(
            Candidate(
                input_index=ordinal + 1,
                line_number=line_number,
                user=user,
                assistant=assistant,
                source_id=source_id,
                category=category,
                source=source,
                metadata=metadata,
            )
        )

    if invalid_rows:
        filter_reasons["JSON无效"] += invalid_rows
        filtered_rows += invalid_rows

    train_count, val_count, component_count = split_candidates(candidates, val_ratio, seed)
    train_rows = [candidate for candidate in candidates if candidate.split == "train"]
    val_rows = [candidate for candidate in candidates if candidate.split == "validation"]
    write_jsonl(train_path, train_rows)
    write_jsonl(validation_path, val_rows)

    all_users = [candidate.user for candidate in candidates]
    all_assistants = [candidate.assistant for candidate in candidates]
    assistant_counts = Counter(norm_key(candidate.assistant) for candidate in candidates)
    assistant_duplicate_extras = sum(count - 1 for count in assistant_counts.values() if count > 1)
    assistant_duplicate_groups = sum(count > 1 for count in assistant_counts.values())

    rng = random.Random(seed)
    preview = list(candidates)
    rng.shuffle(preview)
    preview = [preview_item(candidate) for candidate in preview[:PREVIEW_SIZE]]

    longest = sorted(candidates, key=lambda candidate: (-len(candidate.assistant), candidate.input_index))[:20]
    shortest = sorted(candidates, key=lambda candidate: (len(candidate.assistant), candidate.input_index))[:20]
    report: dict[str, Any] = {
        "name": "QWbot V2 Alpha Qwen3/Qwen4B SFT dataset",
        "version": "qwbot_v2_alpha_dataset_v1",
        "random_seed": seed,
        "validation_ratio": val_ratio,
        "input": {
            "path": str(input_path),
            "metadata_path": str(metadata_path) if metadata_path else None,
            "total_nonempty_lines": total_input,
            "valid_json_objects": len(raw_rows),
            "invalid_rows": invalid_rows,
        },
        "总输入数量": total_input,
        "total_input_count": total_input,
        "最终train数量": train_count,
        "final_train_count": train_count,
        "最终val数量": val_count,
        "final_validation_count": val_count,
        "final_count": len(candidates),
        "类别分布": {
            "all": distribution(candidates, "category"),
            "train": distribution(candidates, "category", "train"),
            "validation": distribution(candidates, "category", "validation"),
        },
        "category_distribution": {
            "all": distribution(candidates, "category"),
            "train": distribution(candidates, "category", "train"),
            "validation": distribution(candidates, "category", "validation"),
        },
        "source分布": {
            "all": distribution(candidates, "source"),
            "train": distribution(candidates, "source", "train"),
            "validation": distribution(candidates, "source", "validation"),
        },
        "source_distribution": {
            "all": distribution(candidates, "source"),
            "train": distribution(candidates, "source", "train"),
            "validation": distribution(candidates, "source", "validation"),
        },
        "平均user长度": round(sum(map(len, all_users)) / len(all_users), 2) if all_users else 0.0,
        "average_user_length": round(sum(map(len, all_users)) / len(all_users), 2) if all_users else 0.0,
        "平均assistant长度": round(sum(map(len, all_assistants)) / len(all_assistants), 2)
        if all_assistants
        else 0.0,
        "average_assistant_length": round(sum(map(len, all_assistants)) / len(all_assistants), 2)
        if all_assistants
        else 0.0,
        "重复率": round(duplicate_pair_count / total_input, 6) if total_input else 0.0,
        "duplicate_rate": round(duplicate_pair_count / total_input, 6) if total_input else 0.0,
        "duplicate_pair_count": duplicate_pair_count,
        "assistant_exact_duplicate_count": assistant_duplicate_extras,
        "assistant_exact_duplicate_rate": round(assistant_duplicate_extras / len(candidates), 6)
        if candidates
        else 0.0,
        "duplicate_statistics": {
            "duplicate_pair_count": duplicate_pair_count,
            "duplicate_pair_rate": round(duplicate_pair_count / total_input, 6) if total_input else 0.0,
            "assistant_exact_duplicate_extra_count": assistant_duplicate_extras,
            "assistant_exact_duplicate_group_count": assistant_duplicate_groups,
            "assistant_exact_duplicate_rate": round(
                assistant_duplicate_extras / len(candidates), 6
            )
            if candidates
            else 0.0,
        },
        "过滤数量": filtered_rows,
        "filtered_count": filtered_rows,
        "过滤原因": {key: int(filter_reasons.get(key, 0)) for key in FILTER_REASON_KEYS},
        "filter_reasons": {key: int(filter_reasons.get(key, 0)) for key in FILTER_REASON_KEYS},
        "metadata_missing_count": metadata_missing_rows,
        "split": {
            "grouping": "root(source_id)-category + exact user-context fingerprint",
            "group_count": component_count,
            "group_leakage_guard": "same group/context component is wholly train or validation",
            "train_group_count": len(
                {root for root, indexes in build_group_components(candidates).items() if candidates[indexes[0]].split == "train"}
            ),
            "validation_group_count": len(
                {root for root, indexes in build_group_components(candidates).items() if candidates[indexes[0]].split == "validation"}
            ),
        },
        "recommended_training_config": {
            "model": "Qwen4B",
            "method": "LoRA",
            "rank": 16,
            "alpha": 32,
            "learning_rate": "1e-5 ~ 2e-5",
            "epoch": "2-3",
            "max_seq_length": 1024,
        },
        "top20_longest_assistant": [length_item(candidate) for candidate in longest],
        "top20_shortest_assistant": [length_item(candidate) for candidate in shortest],
        "random50_preview": preview,
        "top20最长assistant": [length_item(candidate) for candidate in longest],
        "top20最短assistant": [length_item(candidate) for candidate in shortest],
        "随机50条preview": preview,
        "outputs": {
            "train": str(train_path),
            "validation": str(validation_path),
            "report": str(report_path),
        },
        "training_started": False,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--train-output", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--val-output", "--validation-output", dest="validation_output", type=Path, default=DEFAULT_VALIDATION)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--val-ratio", "--validation-ratio", dest="val_ratio", type=float, default=DEFAULT_VAL_RATIO)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = prepare(
        input_path=args.input,
        metadata_path=args.metadata,
        train_path=args.train_output,
        validation_path=args.validation_output,
        report_path=args.report_output,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    print(
        json.dumps(
            {
                "input": report["总输入数量"],
                "train": report["最终train数量"],
                "validation": report["最终val数量"],
                "filtered": report["过滤数量"],
                "average_user_length": report["average_user_length"],
                "average_assistant_length": report["average_assistant_length"],
                "outputs": report["outputs"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
