#!/usr/bin/env python3
"""List/download only JSON annotations from qualialabsAI/SmoothConv.
Network access is only attempted when this script is run; --download is required to save files.
"""
from __future__ import annotations
import argparse, hashlib, json, re, sys
from collections import defaultdict, deque
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen

DATASET = "qualialabsAI/SmoothConv"
API = f"https://huggingface.co/api/datasets/{DATASET}/tree/main?recursive=true"
DEFAULT_OUT = Path("datasets/conversation_raw/smoothconv_json")
SCENARIOS = ("social", "tutoring", "all")
MAX_PAGES = 100
LIST_TIMEOUT = 30
DOWNLOAD_TIMEOUT = 60


def fetch_json(url: str):
    """Fetch one API page, returning its JSON and response headers."""
    req = Request(url, headers={"User-Agent": "QWbot-conversation-pipeline/1.0"})
    with urlopen(req, timeout=LIST_TIMEOUT) as r:
        return json.load(r), r.headers


def _valid_next_url(url: str):
    try:
        parsed = urlparse(url)
        port = parsed.port
    except ValueError as e:
        raise RuntimeError(f"拒绝不安全或跨数据集的分页链接: {url}") from e
    expected_path = f"/api/datasets/{DATASET}/tree/main"
    if (parsed.scheme != "https" or parsed.hostname != "huggingface.co"
            or port is not None or parsed.username or parsed.password
            or parsed.path != expected_path or parsed.fragment):
        raise RuntimeError(f"拒绝不安全或跨数据集的分页链接: {url}")
    return url


def _next_link(headers):
    if headers is None:
        link = ""
    else:
        link = headers.get("Link", headers.get("link", ""))
    # RFC 8288 link-values; the URL itself is taken from the server header.
    for part in link.split(","):
        match = re.search(r"<([^>]+)>\s*;\s*rel\s*=\s*[\"']?next[\"']?", part, re.I)
        if match:
            return match.group(1)
    return None


def _all_pages():
    page_url = API
    seen = set()
    all_entries = []
    for page_number in range(1, MAX_PAGES + 1):
        if page_url in seen:
            raise RuntimeError(f"分页链接循环，拒绝重复页面: {page_url}")
        seen.add(page_url)
        try:
            fetched = fetch_json(page_url)
            # Keep the listing helper compatible with lightweight test doubles
            # that return only the decoded page and no response headers.
            if isinstance(fetched, tuple) and len(fetched) == 2:
                page, headers = fetched
            else:
                page, headers = fetched, None
        except (HTTPError, URLError, TimeoutError, OSError, json.JSONDecodeError) as e:
            raise RuntimeError(f"查询 Hugging Face 清单第 {page_number} 页失败 ({page_url}): {e}") from e
        if not isinstance(page, list):
            raise RuntimeError(f"Hugging Face 清单第 {page_number} 页格式错误（应为数组）")
        all_entries.extend(page)
        raw_next = _next_link(headers)
        if not raw_next:
            return all_entries
        next_url = _valid_next_url(urljoin(page_url, raw_next))
        if next_url in seen:
            raise RuntimeError(f"分页链接循环，拒绝重复页面: {next_url}")
        page_url = next_url
    raise RuntimeError(f"分页超过最大页数 {MAX_PAGES}，拒绝返回不完整清单")

def entries():
    try:
        data = _all_pages()
    except RuntimeError:
        raise
    result = []
    seen_paths = set()
    for x in data:
        path = x.get("path", "")
        # The repository contains media/model artifacts too.  Only annotation JSON
        # files are candidates; filtering by extension also prevents downloading them.
        if (x.get("type") == "file" and path.lower().endswith(".json")
                and scenario_for_path(path) in {"social", "tutoring"}):
            if path in seen_paths:
                continue
            seen_paths.add(path)
            # API size may be absent for LFS entries; unknown sizes are handled conservatively.
            result.append({"path": path, "size": x.get("size"), "oid": x.get("oid")})
    return result

def scenario_for_path(path: str):
    """Classify only from the actual first-level directory name."""
    normalized = path.replace("\\", "/")
    if "/" not in normalized:
        return None
    first = normalized.split("/", 1)[0].lower()
    if first.startswith("none_edu"):
        return "social"
    if first.startswith("edu"):
        return "tutoring"
    return None

def score(e):
    return e["path"].lower()


def recording_key(path):
    """Group adjacent segment files belonging to one original recording."""
    filename = path.replace("\\", "/").rsplit("/", 1)[-1].lower()
    return re.sub(r"_seg[0-9][0-9_]*", "", filename)

def select_entries(all_files, scenario, max_files, max_bytes):
    candidates = [e for e in all_files
                  if e["path"].lower().endswith(".json")
                  and scenario_for_path(e["path"]) in {"social", "tutoring"}
                  and (scenario == "all" or scenario_for_path(e["path"]) == scenario)]
    groups = defaultdict(deque)
    for e in sorted(candidates, key=score):
        groups[recording_key(e["path"])].append(e)
    selected, total = [], 0
    # Round-robin across recordings before taking another segment from one recording.
    while groups and len(selected) < max_files:
        for key in list(sorted(groups)):
            queue = groups[key]
            e = queue.popleft()
            size = e.get("size")
            if size is not None and total + size <= max_bytes:
                selected.append(e); total += size
            if not queue:
                del groups[key]
            if len(selected) >= max_files:
                break
    return candidates, selected, total

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--list-only", action="store_true", help="只查询并列出候选文件（默认行为）")
    ap.add_argument("--download", action="store_true", help="明确开启 JSON 下载；不会下载音频或模型")
    ap.add_argument("--scenario", choices=SCENARIOS, default="social", help="数据场景（默认 social）")
    ap.add_argument("--max-files", type=int, default=30)
    ap.add_argument("--max-total-mb", type=float, default=20.0)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args(argv)
    if args.max_files <= 0 or args.max_total_mb <= 0:
        ap.error("--max-files 和 --max-total-mb 必须为正数")
    if args.download and args.list_only:
        ap.error("--download 与 --list-only 不能同时使用")
    try:
        all_files = entries()
    except RuntimeError as e:
        print(f"错误: {e}", file=sys.stderr); return 2
    counts = {s: sum(e["path"].lower().endswith(".json") and scenario_for_path(e["path"]) == s
                    for e in all_files) for s in ("social", "tutoring")}
    print(f"可用文件数量: social={counts['social']}，tutoring={counts['tutoring']}")
    if args.scenario == "social" and counts["social"] == 0:
        print("错误: social 场景没有可用 JSON 文件；已停止，不会补充 tutoring 文件。", file=sys.stderr)
        return 5
    candidates, selected, total = select_entries(all_files, args.scenario, args.max_files, args.max_total_mb * 1024 * 1024)
    print(f"来源: {DATASET} ({API})")
    print(f"{args.scenario} 候选 JSON: {len(candidates)}；选中: {len(selected)}；估计体积: {total / 1024 / 1024:.2f} MB")
    for i, e in enumerate(selected, 1): print(f"{i:02d}  {e['size'] or 0:>10}  {e['path']}")
    if not args.download:
        print("未下载。需要下载时请显式追加 --download。")
        return 0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    downloaded_total = 0
    max_total_bytes = int(args.max_total_mb * 1024 * 1024)
    for e in selected:
        path = e["path"]
        url = f"https://huggingface.co/datasets/{DATASET}/resolve/main/{path}"
        dest = args.output_dir / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            req = Request(url, headers={"User-Agent": "QWbot-conversation-pipeline/1.0"})
            with urlopen(req, timeout=DOWNLOAD_TIMEOUT) as r: blob = r.read()
        except (HTTPError, URLError, TimeoutError, OSError) as ex:
            print(f"错误: 下载失败 {path}: {ex}", file=sys.stderr); return 3
        if downloaded_total + len(blob) > max_total_bytes:
            print(f"错误: 下载内容超过 --max-total-mb 限制，拒绝写入 {path}", file=sys.stderr)
            return 6
        downloaded_total += len(blob)
        digest = hashlib.sha256(blob).hexdigest()
        if dest.exists():
            old = hashlib.sha256(dest.read_bytes()).hexdigest()
            if old != digest:
                print(f"错误: 目标已存在且 SHA256 不同，拒绝覆盖: {dest}", file=sys.stderr); return 4
            print(f"跳过已存在且 SHA256 相同: {dest}")
        else: dest.write_bytes(blob)
        dest.with_suffix(dest.suffix + ".sha256").write_text(digest + "  " + dest.name + "\n", encoding="utf-8")
        meta = {"source_dataset": DATASET, "source_path": path, "source_url": url, "sha256": digest, "bytes": len(blob), "license": "CC BY-NC 4.0", "downloaded_by": "01_fetch_smoothconv_json.py"}
        dest.with_suffix(dest.suffix + ".source.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        manifest.append(meta)
    (args.output_dir / "download_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"已下载 {len(manifest)} 个 JSON；未下载任何音频或模型文件。")
    return 0

if __name__ == "__main__": raise SystemExit(main())
