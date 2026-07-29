#!/usr/bin/env python3
"""
Stage 1: SAM2 Segmentation

1. For each video, extract ONLY the first frame
2. Run SAM2 automatic mask generation to segment all objects
3. Filter meaningless fragments (small area, low stability)
4. Save masked crops

Output: objects/crops/  +  objects/manifests/segments.json

Usage:
    conda activate robocoin-object
    python stage1_segment.py                  # all videos
    python stage1_segment.py --limit 10       # first 10
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch

BASE_DIR = Path(__file__).resolve().parent
VIDEO_ROOT = BASE_DIR / "RoboCOIN_datasets"
CROPS_DIR = BASE_DIR / "objects" / "crops"
MANIFEST_DIR = BASE_DIR / "objects" / "manifests"

SAM2_VARIANT = "tiny"
SAM2_CHECKPOINT = None

# Filters
PRED_IOU_THRESH = 0.88
STABILITY_THRESH = 0.95
MIN_MASK_AREA_RATIO = 0.005   # >0.5% of image
MAX_MASKS_PER_FRAME = 20


def find_videos() -> list[Path]:
    videos = []
    for p in sorted(VIDEO_ROOT.rglob("episode_000000.mp4")):
        if "wrist" in str(p) or "chunk-000" not in str(p):
            continue
        videos.append(p)
    return videos


def get_first_frame(video_path: Path) -> np.ndarray | None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


def load_sam2_generator():
    from sam2.build_sam import load_model
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading SAM2 ({SAM2_VARIANT}) on {device}…", flush=True)

    model = load_model(
        variant=SAM2_VARIANT,
        ckpt_path=SAM2_CHECKPOINT,
        device=device,
    )
    return SAM2AutomaticMaskGenerator(
        model=model,
        pred_iou_thresh=PRED_IOU_THRESH,
        stability_score_thresh=STABILITY_THRESH,
        min_mask_region_area=100,
        box_nms_thresh=0.7,
        output_mode="binary_mask",
    )


def segment_frame(frame_bgr: np.ndarray, generator) -> list[dict]:
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w = frame_rgb.shape[:2]
    min_area = int(h * w * MIN_MASK_AREA_RATIO)

    masks = generator.generate(frame_rgb)

    results = []
    for m in masks:
        area = m["area"]
        if area < min_area:
            continue
        results.append({
            "segmentation": m["segmentation"],
            "bbox": [int(v) for v in m["bbox"]],
            "area": int(area),
            "stability_score": round(m["stability_score"], 4),
            "predicted_iou": round(m["predicted_iou"], 4),
        })

    results.sort(key=lambda x: x["stability_score"], reverse=True)
    return results[:MAX_MASKS_PER_FRAME]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    videos = find_videos()
    print(f"Found {len(videos)} videos", flush=True)
    if args.limit:
        videos = videos[:args.limit]

    generator = load_sam2_generator()
    CROPS_DIR.mkdir(parents=True, exist_ok=True)

    all_crops = []
    stats = {"videos": 0, "segments": 0}
    t0 = time.time()

    for i, vpath in enumerate(videos, 1):
        rel = vpath.relative_to(VIDEO_ROOT)
        dataset, camera = rel.parts[0], rel.parts[3]

        frame = get_first_frame(vpath)
        if frame is None:
            print(f"  SKIP {rel}: cannot read", flush=True)
            continue

        masks = segment_frame(frame, generator)
        stats["videos"] += 1
        stats["segments"] += len(masks)

        crop_dir = CROPS_DIR / dataset / camera
        crop_dir.mkdir(parents=True, exist_ok=True)

        for j, mask in enumerate(masks):
            x1, y1, x2, y2 = mask["bbox"]
            h, w = frame.shape[:2]

            pad_w = int((x2 - x1) * 0.05)
            pad_h = int((y2 - y1) * 0.05)
            cx1, cy1 = max(0, x1 - pad_w), max(0, y1 - pad_h)
            cx2, cy2 = min(w, x2 + pad_w), min(h, y2 + pad_h)

            crop = frame[cy1:cy2, cx1:cx2]
            if crop.size == 0:
                continue

            mask_roi = mask["segmentation"][cy1:cy2, cx1:cx2]
            crop_masked = crop.copy()
            crop_masked[~mask_roi] = 0

            crop_name = f"seg_{j:02d}.jpg"
            crop_path = crop_dir / crop_name
            cv2.imwrite(str(crop_path), crop_masked)

            all_crops.append({
                "dataset": dataset,
                "camera": camera,
                "video": str(rel),
                "seg_idx": j,
                "bbox": [cx1, cy1, cx2, cy2],
                "area": mask["area"],
                "stability_score": mask["stability_score"],
                "predicted_iou": mask["predicted_iou"],
                "crop_path": str(crop_path.relative_to(BASE_DIR)),
            })

        elapsed = time.time() - t0
        print(
            f"\r[{i:4d}/{len(videos)}] segs={stats['segments']} "
            f"({i / elapsed:.1f}/s)", end="", flush=True,
        )

    manifest_path = MANIFEST_DIR / "segments.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(all_crops, f, indent=2)

    print(f"\n\nDone: {stats['videos']} videos, "
          f"{stats['segments']} segments, {len(all_crops)} crops")
    print(f"Crops:    {CROPS_DIR}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
