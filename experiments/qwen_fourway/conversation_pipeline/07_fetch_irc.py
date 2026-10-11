#!/usr/bin/env python3
"""Inspect/download a small, explicitly selected IRC Disentanglement file set.

Default mode only checks/prints the allowlisted URLs.  --download is required
for network access and writes never overwrite existing files.
"""
from __future__ import annotations
import argparse, hashlib, json, re, sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

REPOSITORY = "jkkummerfeld/irc-disentanglement"
RAW_ROOT = "https://raw.githubusercontent.com/jkkummerfeld/irc-disentanglement/master/"
DEFAULT_RAW = Path("datasets/conversation_raw/irc_disentanglement")
DEFAULT_MAX_FILES = 6
DEFAULT_MAX_BYTES = 50 * 1024 * 1024
ALLOWED = re.compile(r"^(?:.+)\.(?:raw|ascii|annotation)\.txt$")
TIMEOUT = 30

def official_url(url: str) -> bool:
    return url.startswith(RAW_ROOT) and bool(ALLOWED.match(url[len(RAW_ROOT):])) and ".." not in url[len(RAW_ROOT):].split("/")

def url_for(path: str) -> str:
    path = path.lstrip("/")
    if not ALLOWED.match(path) or ".." in path.split("/"):
        raise ValueError("只允许官方仓库中的 .raw.txt/.ascii.txt/.annotation.txt 文件")
    return RAW_ROOT + path

def inspect_sources(paths):
    urls = [url_for(p) for p in paths]
    if len(set(urls)) != len(urls): raise ValueError("来源文件重复")
    return [{"path": p, "url": u, "official": official_url(u)} for p, u in zip(paths, urls)]

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--file", action="append", dest="files", help="仓库内相对路径；重复使用以选择 raw/ascii/annotation 配套文件")
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_RAW)
    ap.add_argument("--max-files", type=int, default=DEFAULT_MAX_FILES)
    ap.add_argument("--max-total-mb", type=float, default=50)
    args = ap.parse_args(argv)
    paths = args.files or []
    if not paths:
        print("未指定文件。请从官方仓库文档选择少量配套 .raw.txt/.ascii.txt/.annotation.txt 文件，再用 --file 重试。")
        return 0
    if not (0 < args.max_files <= DEFAULT_MAX_FILES): ap.error(f"--max-files 必须在 1 到 {DEFAULT_MAX_FILES} 之间")
    if len(paths) > args.max_files: ap.error("文件数量超过限制")
    limit = int(args.max_total_mb * 1024 * 1024)
    if limit <= 0 or limit > DEFAULT_MAX_BYTES: ap.error(f"总大小必须在 0 和 {DEFAULT_MAX_BYTES // 1024 // 1024} MiB 之间")
    try: entries = inspect_sources(paths)
    except ValueError as exc: print(f"来源检查失败: {exc}", file=sys.stderr); return 2
    print(json.dumps({"repository": REPOSITORY, "files": entries, "max_total_bytes": limit}, ensure_ascii=False, indent=2))
    if not args.download:
        print("仅检查白名单来源；未下载。需要下载时请显式追加 --download。"); return 0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    blobs=[]; total=0
    try:
        for e in entries:
            req=Request(e["url"], headers={"User-Agent":"QWbot-IRC-pipeline/1.0"})
            with urlopen(req, timeout=TIMEOUT) as resp:
                blob=resp.read(limit-total+1)
            total += len(blob)
            if total > limit: raise ValueError("下载内容超过 --max-total-mb")
            blobs.append((e, blob))
    except (HTTPError, URLError, TimeoutError, OSError, ValueError) as exc:
        print(f"下载失败: {exc}", file=sys.stderr); return 3
    for e, blob in blobs:
        dest=args.output_dir / Path(e["path"]).name
        if dest.exists(): print(f"错误: 拒绝覆盖已有文件: {dest}", file=sys.stderr); return 4
        digest=hashlib.sha256(blob).hexdigest(); dest.write_bytes(blob)
        dest.with_suffix(dest.suffix + ".sha256").write_text(f"{digest}  {dest.name}\n", encoding="utf-8")
        meta={"repository":REPOSITORY,"source_url":e["url"],"source_path":e["path"],"sha256":digest,"bytes":len(blob),"license_status":"pending_confirmation","downloaded_by":"07_fetch_irc.py"}
        dest.with_suffix(dest.suffix + ".source.json").write_text(json.dumps(meta,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
        print(f"已保存 {dest} ({len(blob)} bytes)")
    return 0
if __name__ == "__main__": raise SystemExit(main())
