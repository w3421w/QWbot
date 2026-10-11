#!/usr/bin/env python3
"""Inspect/download the official Dialogue-MPDD mpdd.zip safely.

The default action performs a HEAD/source-size check only.  Network access and
writing the archive require the explicit --download flag.
"""
from __future__ import annotations
import argparse, hashlib, json, sys, zipfile
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

REPOSITORY = "ntunlplab/Dialogue-MPDD"
SOURCE_URL = "https://raw.githubusercontent.com/ntunlplab/Dialogue-MPDD/main/mpdd.zip"
DEFAULT_RAW = Path("datasets/conversation_raw/mpdd")
DEFAULT_MAX_BYTES = 512 * 1024 * 1024
TIMEOUT = 30


def _official_url(url: str) -> bool:
    return url == SOURCE_URL


def inspect_source(url: str = SOURCE_URL):
    if not _official_url(url):
        raise ValueError("只允许官方 GitHub 仓库的固定 mpdd.zip 来源")
    req = Request(url, method="HEAD", headers={"User-Agent": "QWbot-MPDD-pipeline/1.0"})
    with urlopen(req, timeout=TIMEOUT) as response:
        size = response.headers.get("Content-Length")
        return {"url": url, "status": getattr(response, "status", 200),
                "content_length": int(size) if size and size.isdigit() else None,
                "content_type": response.headers.get("Content-Type", "")}


def zip_member_is_safe(name: str) -> bool:
    """Reject absolute paths, traversal and Windows drive paths."""
    p = name.replace("\\", "/")
    if not p or p.startswith("/") or p.startswith("\\") or ":" in p.split("/", 1)[0]:
        return False
    parts = [x for x in p.split("/") if x not in ("", ".")]
    return ".." not in parts


def validate_zip_bytes(blob: bytes, max_uncompressed_bytes: int = DEFAULT_MAX_BYTES,
                       max_single_bytes: int = DEFAULT_MAX_BYTES):
    import stat
    total = 0
    with zipfile.ZipFile(__import__("io").BytesIO(blob)) as archive:
        for info in archive.infolist():
            if not zip_member_is_safe(info.filename):
                raise ValueError(f"ZIP 路径穿越或绝对路径: {info.filename}")
            mode = (info.external_attr >> 16) & 0o170000
            if mode == stat.S_IFLNK:
                raise ValueError(f"ZIP 符号链接被拒绝: {info.filename}")
            if info.file_size > max_single_bytes:
                raise ValueError(f"ZIP 单文件大小超过限制: {info.filename}")
            total += info.file_size
            if total > max_uncompressed_bytes:
                raise ValueError("ZIP 解压总大小超过限制")
            if info.is_dir():
                continue
            # Check compression ratio as a zip bomb guard when metadata is present.
            if info.compress_size == 0 and info.file_size > 0:
                raise ValueError(f"ZIP 压缩大小异常: {info.filename}")
            if info.compress_size and info.file_size / info.compress_size > 1000:
                raise ValueError(f"ZIP 压缩比异常: {info.filename}")
    return total


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--download", action="store_true", help="明确下载并保存 mpdd.zip")
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_RAW)
    ap.add_argument("--max-total-mb", type=float, default=512)
    args = ap.parse_args(argv)
    if args.max_total_mb <= 0:
        ap.error("--max-total-mb 必须为正数")
    limit = int(args.max_total_mb * 1024 * 1024)
    try:
        info = inspect_source()
    except (HTTPError, URLError, TimeoutError, OSError, ValueError) as exc:
        print(f"来源检查失败: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"repository": REPOSITORY, **info, "max_bytes": limit}, ensure_ascii=False, indent=2))
    if info.get("content_length") and info["content_length"] > limit:
        print("错误: 来源文件超过总下载大小限制。", file=sys.stderr)
        return 6
    if not args.download:
        print("仅检查来源和文件大小；未下载。需要下载时请显式追加 --download。")
        return 0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dest = args.output_dir / "mpdd.zip"
    if dest.exists():
        print(f"错误: 目标已存在，拒绝覆盖: {dest}", file=sys.stderr)
        return 4
    req = Request(SOURCE_URL, headers={"User-Agent": "QWbot-MPDD-pipeline/1.0"})
    try:
        with urlopen(req, timeout=TIMEOUT) as response:
            chunks, total = [], 0
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk: break
                total += len(chunk)
                if total > limit: raise ValueError("下载内容超过 --max-total-mb")
                chunks.append(chunk)
        blob = b"".join(chunks)
        validate_zip_bytes(blob, limit)
    except (HTTPError, URLError, TimeoutError, OSError, ValueError, zipfile.BadZipFile) as exc:
        print(f"下载或 ZIP 安全检查失败: {exc}", file=sys.stderr)
        return 3
    digest = hashlib.sha256(blob).hexdigest()
    dest.write_bytes(blob)
    dest.with_suffix(".zip.sha256").write_text(f"{digest}  {dest.name}\n", encoding="utf-8")
    metadata = {"repository": REPOSITORY, "source_url": SOURCE_URL, "sha256": digest,
                "bytes": len(blob), "license_status": "pending_confirmation",
                "downloaded_by": "04_fetch_mpdd.py"}
    dest.with_suffix(".zip.source.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"已保存 {dest} ({len(blob)} bytes)，未解压。")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
