#!/usr/bin/env python3
"""
Download the meta/ directory from all datasets under the RoboCOIN
organization on ModelScope.

Usage:
    conda activate robocoin
    python download_meta.py

Or dry-run first:
    python download_meta.py --dry-run
"""

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from modelscope.hub.api import HubApi
from modelscope.hub.snapshot_download import snapshot_download

# ── config ──────────────────────────────────────────────────────────
ORG = "RoboCOIN"
BASE_DIR = Path(__file__).resolve().parent / "RoboCOIN_datasets"
MAX_WORKERS = 15  # parallel downloads
RETRY_COUNT = 3
RETRY_DELAY = 2  # seconds
# ─────────────────────────────────────────────────────────────────────


def list_datasets() -> list[str]:
    """Return all repo_ids under the RoboCOIN org (handles pagination)."""
    api = HubApi()
    dataset_ids = []
    page = 1
    while True:
        result = api.list_repos("dataset", owner=ORG, page_number=page, page_size=50)
        for r in result:
            dataset_ids.append(f"{ORG}/{r.name}")
        if not result.has_next:
            break
        page += 1
    return dataset_ids


def download_meta(repo_id: str) -> tuple[str, str]:
    """Download meta/ for one dataset. Returns (repo_id, status)."""
    name = repo_id.split("/", 1)[1]
    target = str(BASE_DIR / name)

    meta_dir = Path(target) / "meta"
    if meta_dir.is_dir() and any(meta_dir.iterdir()):
        return repo_id, "skipped"

    for attempt in range(1, RETRY_COUNT + 1):
        try:
            snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                allow_patterns=["meta/**"],
                local_dir=target,
            )
            return repo_id, "done"
        except Exception as exc:
            if attempt < RETRY_COUNT:
                time.sleep(RETRY_DELAY)
            else:
                return repo_id, f"failed: {exc}"


def main():
    parser = argparse.ArgumentParser(description="Download meta/ from RoboCOIN datasets")
    parser.add_argument("--dry-run", action="store_true", help="List datasets only")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--limit", type=int, default=0, help="Limit to first N datasets")
    args = parser.parse_args()

    print("Listing datasets…", flush=True)
    datasets = list_datasets()
    print(f"Found {len(datasets)} datasets", flush=True)

    if args.limit:
        datasets = datasets[: args.limit]
        print(f"Limited to first {args.limit}", flush=True)

    if args.dry_run:
        for d in datasets:
            print(f"  {d}")
        return

    BASE_DIR.mkdir(parents=True, exist_ok=True)

    stats = {"done": 0, "skipped": 0, "failed": 0}
    failed = []

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(download_meta, rid): rid for rid in datasets}
        for i, fut in enumerate(as_completed(futures), 1):
            repo_id, status = fut.result()
            if status == "done":
                stats["done"] += 1
            elif status == "skipped":
                stats["skipped"] += 1
            else:
                stats["failed"] += 1
                failed.append((repo_id, status))

            elapsed = time.time() - t0
            rate = i / elapsed if elapsed > 0 else 0
            print(
                f"\r[{i:4d}/{len(datasets)}] "
                f"done={stats['done']} skipped={stats['skipped']} "
                f"failed={stats['failed']}  "
                f"({rate:.1f}/s)",
                end="",
                flush=True,
            )

    print("\n\n=== Summary ===")
    print(f"Total:   {len(datasets)}")
    print(f"Done:    {stats['done']}")
    print(f"Skipped: {stats['skipped']}")
    print(f"Failed:  {stats['failed']}")
    print(f"Time:    {time.time() - t0:.0f}s")

    if failed:
        print("\nFailed datasets:")
        for rid, reason in failed:
            print(f"  {rid}: {reason}")


if __name__ == "__main__":
    main()
