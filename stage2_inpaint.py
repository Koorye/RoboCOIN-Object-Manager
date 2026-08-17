#!/usr/bin/env python3
"""
Stage 2: Inpaint Occluded Object Regions with Moebius

Uses Moebius (0.22B, ECCV 2026) to fill occluded regions:
  https://github.com/hustvl/Moebius

For each crop: object = non-black pixels, background = black → inpaint.

Input:  objects/crops/<dataset>/<camera>/seg_*_crop.jpg
Output: seg_*_inpaint.jpg

Usage:
    pip install moebius-inpainting
    python stage2_inpaint.py
    python stage2_inpaint.py --limit 50
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent
CROPS_DIR = BASE_DIR / "objects" / "crops"

OCCLUSION_RATIO = 0.85


def find_crop_dirs() -> list[Path]:
    dirs = set()
    for p in CROPS_DIR.rglob("seg_00_mask.png"):
        dirs.add(p.parent)
    return sorted(dirs)


def load_moebius():
    """Load Moebius inpainting model."""
    from moebius import MoebiusPipeline
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    print(f"Loading Moebius on {device}…", flush=True)
    pipe = MoebiusPipeline.from_pretrained(
        "hustvl/Moebius",
        torch_dtype=dtype,
    ).to(device)
    return pipe


def needs_inpaint(crop_path: Path) -> bool:
    crop = np.array(Image.open(crop_path).convert("RGB"))
    h, w = crop.shape[:2]
    obj_pixels = (crop.sum(axis=-1) > 10).sum()
    return obj_pixels / (h * w) < OCCLUSION_RATIO


def inpaint_crop(crop_path: Path, pipe) -> Image.Image | None:
    """Run Moebius on a crop. Returns inpainted PIL Image."""
    crop = Image.open(crop_path).convert("RGB")
    arr = np.array(crop)
    obj = arr.sum(axis=-1) > 10
    if obj.sum() < 100:
        return None

    hole = ~obj
    if hole.sum() < obj.sum() * 0.02:
        return None

    mask = Image.fromarray((hole * 255).astype(np.uint8))

    result = pipe(image=crop, mask=mask, num_inference_steps=20)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    # Clean previous
    old_count = 0
    for old in CROPS_DIR.rglob("*_inpaint.jpg"):
        old.unlink()
        old_count += 1
    if old_count:
        print(f"Removed {old_count} previous inpaint results")

    pipe = load_moebius()
    dirs = find_crop_dirs()
    print(f"Found {len(dirs)} crop directories", flush=True)
    if args.limit:
        dirs = dirs[: args.limit]

    stats = {"needed": 0, "inpainted": 0}
    manifest_dir = BASE_DIR / "objects" / "manifests"

    for d in tqdm(dirs, desc="Inpainting"):
        manifest_path = manifest_dir / d.relative_to(CROPS_DIR) / "detections.jsonl"
        if not manifest_path.exists():
            continue

        with open(manifest_path) as f:
            items = [json.loads(line) for line in f if line.strip()]

        for item in items:
            crop_p = BASE_DIR / item["crop_path"]
            if not crop_p.exists() or not needs_inpaint(crop_p):
                continue

            stats["needed"] += 1
            result = inpaint_crop(crop_p, pipe)

            if result is not None:
                out_path = d / f"seg_{item['seg_idx']:02d}_inpaint.jpg"
                result.save(str(out_path))
                stats["inpainted"] += 1

    print(f"\nDone:")
    print(f"  Needed inpainting: {stats['needed']}")
    print(f"  Inpainted:         {stats['inpainted']}")


if __name__ == "__main__":
    main()
