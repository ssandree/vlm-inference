#!/usr/bin/env python3
"""Resume-safe ranged download of Qwen3-VL-30B-A3B-Instruct BF16 weights.

A full-size file is accepted only when its tail is not empty and, when the
Hub etag is a sha256, the digest matches. Incomplete preallocated files are
deleted and fetched again.
"""

from __future__ import annotations

import hashlib
import os
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from huggingface_hub import HfApi, get_hf_file_metadata, hf_hub_url

REPO = "Qwen/Qwen3-VL-30B-A3B-Instruct"
REV = "9c4b90e1e4ba969fd3b5378b57d966d725f1b86c"
DEST = Path("/data1/vailab02_dir/vlm_models/Qwen3-VL-30B-A3B-Instruct")
WORKERS = 8
CHUNK = 8 * 1024 * 1024


def tail_is_empty(path: Path) -> bool:
    size = path.stat().st_size
    with path.open("rb") as handle:
        handle.seek(max(0, size - 1024 * 1024))
        blob = handle.read()
    return len(blob) >= 1024 * 1024 and blob.count(0) / len(blob) > 0.99


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch_range(url: str, start: int, end: int) -> bytes:
    last: Exception | None = None
    for attempt in range(6):
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(req, timeout=180) as resp:
                data = resp.read()
            if len(data) != end - start + 1:
                raise OSError(f"short read {len(data)} != {end - start + 1}")
            return data
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(min(20, 2 ** attempt))
    raise RuntimeError(f"{start}-{end} failed: {last}")


def download_one(name: str, size: int, etag: str | None) -> None:
    dest = DEST / name
    part = Path(str(dest) + ".part")
    if dest.is_file() and dest.stat().st_size == size and not tail_is_empty(dest):
        if etag and len(etag) == 64:
            got = sha256_file(dest)
            if got != etag:
                print(f"BADHASH {name} {got} != {etag}", flush=True)
                dest.unlink()
            else:
                print(f"SKIP {name}", flush=True)
                return
        else:
            print(f"SKIP {name}", flush=True)
            return
    if dest.is_file():
        dest.unlink()
    if part.is_file() and (part.stat().st_size > size or tail_is_empty(part)):
        part.unlink()
    url = f"https://huggingface.co/{REPO}/resolve/{REV}/{name}"
    offset = part.stat().st_size if part.is_file() else 0
    with part.open("ab") as handle:
        handle.truncate(size)
    ranges = []
    cursor = offset
    while cursor < size:
        end = min(size - 1, cursor + CHUNK - 1)
        ranges.append((cursor, end))
        cursor = end + 1
    print(f"GET {name} from {offset} ranges {len(ranges)}", flush=True)
    done = 0
    lock = threading.Lock()
    t0 = time.time()

    def work(span: tuple[int, int]) -> int:
        start, end = span
        data = fetch_range(url, start, end)
        with part.open("r+b") as handle:
            handle.seek(start)
            handle.write(data)
        return len(data)

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = [pool.submit(work, span) for span in ranges]
        for fut in as_completed(futures):
            n = fut.result()
            with lock:
                done += n
                if done // (32 * 1024 * 1024) != (done - n) // (32 * 1024 * 1024):
                    rate = done / max(time.time() - t0, 1e-3) / 1024 / 1024
                    print(f"  {name} {(offset + done) / 1024**3:.2f}/{size / 1024**3:.2f} GiB {rate:.2f} MiB/s", flush=True)
    if tail_is_empty(part):
        raise RuntimeError(f"{name} tail is empty after download")
    os.replace(part, dest)
    if etag and len(etag) == 64:
        got = sha256_file(dest)
        if got != etag:
            dest.unlink()
            raise RuntimeError(f"hash mismatch {name}")
    print(f"OK {name}", flush=True)


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    entries = [e for e in api.list_repo_tree(REPO, revision=REV, recursive=True) if getattr(e, "size", None)]
    # Large shards first so a failure is obvious, but keep config/tokenizer available early.
    entries.sort(key=lambda e: (0 if str(e.path).endswith(".safetensors") else 1, e.path))
    # Tokenizer files are needed before weights for nothing here; download small files first.
    entries.sort(key=lambda e: (0 if not str(e.path).endswith(".safetensors") else 1, e.path))
    for entry in entries:
        url = hf_hub_url(REPO, entry.path, revision=REV)
        meta = get_hf_file_metadata(url)
        etag = (meta.etag or "").strip('"')
        download_one(entry.path, int(entry.size), etag if len(etag) == 64 else None)
    print("DOWNLOAD_DONE", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(f"DOWNLOAD_BAD {type(exc).__name__}: {exc}", flush=True)
        sys.exit(1)
