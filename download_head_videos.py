#!/usr/bin/env python3
"""
Download videos/chunk-000/*/episode_000000.mp4 (first episode, all cameras
except wrist) from ALL RoboCOIN datasets.

Usage:
    conda activate robocoin
    python download_head_videos.py
    python download_head_videos.py --limit 5   # test first N
"""

import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from modelscope.hub.api import HubApi
from modelscope.hub.snapshot_download import snapshot_download

ORG = "RoboCOIN"
BASE_DIR = Path(__file__).resolve().parent / "RoboCOIN_datasets"
ALLOW_PATTERNS = ["videos/chunk-000/*/episode_000000.mp4"]
IGNORE_PATTERNS = ["videos/chunk-000/*wrist*/episode_000000.mp4"]
MAX_WORKERS = 10
RETRY_COUNT = 3
RETRY_DELAY = 3


def list_datasets() -> list[str]:
    api = HubApi()
    ids = []
    page = 1
    while True:
        result = api.list_repos("dataset", owner=ORG, page_number=page, page_size=50)
        for r in result:
            ids.append(f"{ORG}/{r.name}")
        if not result.has_next:
            break
        page += 1
    return ids


def download_one(repo_id: str) -> tuple[str, str]:
    """Try to download. Returns (repo_id, status)."""
    name = repo_id.split("/", 1)[1]
    target = str(BASE_DIR / name)

    # Check if already downloaded (any non-wrist camera with episode_000000.mp4)
    video_dir = Path(target) / "videos" / "chunk-000"
    if video_dir.is_dir():
        for cam_dir in video_dir.iterdir():
            if cam_dir.is_dir() and "wrist" not in cam_dir.name:
                ep = cam_dir / "episode_000000.mp4"
                if ep.exists() and ep.stat().st_size > 0:
                    return repo_id, "skipped"

    for attempt in range(1, RETRY_COUNT + 1):
        try:
            snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                allow_patterns=ALLOW_PATTERNS,
                ignore_patterns=IGNORE_PATTERNS,
                local_dir=target,
            )
            # Verify download actually happened
            video_dir = Path(target) / "videos" / "chunk-000"
            if video_dir.is_dir():
                for cam_dir in video_dir.iterdir():
                    if cam_dir.is_dir() and "wrist" not in cam_dir.name:
                        ep = cam_dir / "episode_000000.mp4"
                        if ep.exists() and ep.stat().st_size > 0:
                            return repo_id, "done"
            return repo_id, "no_match"
        except Exception as exc:
            if attempt < RETRY_COUNT:
                time.sleep(RETRY_DELAY)
            else:
                return repo_id, f"failed: {exc}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    args = parser.parse_args()

    print("Listing datasets…", flush=True)
    all_ds = list_datasets()
    print(f"Total: {len(all_ds)}", flush=True)

    if args.limit:
        all_ds = all_ds[: args.limit]
        print(f"Limited to {args.limit}", flush=True)

    BASE_DIR.mkdir(parents=True, exist_ok=True)

    stats = {"done": 0, "skipped": 0, "no_match": 0, "failed": 0}
    failed = []
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(download_one, ds): ds for ds in all_ds}
        for i, fut in enumerate(as_completed(futures), 1):
            repo_id, status = fut.result()
            if status == "done":
                stats["done"] += 1
            elif status == "skipped":
                stats["skipped"] += 1
            elif status == "no_match":
                stats["no_match"] += 1
            else:
                stats["failed"] += 1
                failed.append((repo_id, status))

            elapsed = time.time() - t0
            rate = i / elapsed if elapsed > 0 else 0
            print(
                f"\r[{i:4d}/{len(all_ds)}] done={stats['done']} "
                f"skipped={stats['skipped']} no_match={stats['no_match']} "
                f"failed={stats['failed']} ({rate:.1f}/s)",
                end="",
                flush=True,
            )

    print("\n\n=== Summary ===")
    print(f"Total:     {len(all_ds)}")
    print(f"Done:      {stats['done']}")
    print(f"Skipped:   {stats['skipped']}")
    print(f"No match:  {stats['no_match']}")
    print(f"Failed:    {stats['failed']}")
    print(f"Time:      {time.time() - t0:.0f}s")

    if failed:
        print("\nFailed:")
        for rid, reason in failed:
            print(f"  {rid}: {reason}")


if __name__ == "__main__":
    main()
