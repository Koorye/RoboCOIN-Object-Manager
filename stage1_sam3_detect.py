#!/usr/bin/env python3
"""
Stage 1: SAM3 Object Detection

For each non-wrist video:
  1. Extract FIRST frame via ffmpeg (supports AV1/h264)
  2. Run SAM3 with comprehensive text prompts to detect all objects
  3. Save: {crop}.jpg (masked), {crop}_mask.png, metadata.jsonl

Usage:
    python stage1_sam3_detect.py
    python stage1_sam3_detect.py --limit 10
    python stage1_sam3_detect.py --gpu 0

Requirements (server):
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cuXXX
    pip install sam3 einops timm pillow numpy tqdm
    pip install opencv-python  # optional, ffmpeg fallback used for AV1

Checkpoints:
    Put sam3.pt in ./sam3_weights/ or set SAM3_CKPT env var.
    BPE vocab: ./sam3_weights/bpe_simple_vocab_16e6.txt.gz
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# ── Paths ───────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
VIDEO_ROOT = BASE_DIR / "RoboCOIN_datasets"
OUTPUT_DIR = BASE_DIR / "objects"
CROPS_DIR = OUTPUT_DIR / "crops"
MANIFEST_DIR = OUTPUT_DIR / "manifests"

SAM3_REPO = BASE_DIR / "sam3"
SAM3_CKPT = Path(
    os.environ.get("SAM3_CKPT", str(BASE_DIR / "checkpoints" / "sam3" / "sam3.pt"))
)
BPE_PATH = Path(
    os.environ.get(
        "BPE_PATH",
        str(SAM3_REPO / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz"),
    )
)

# ── Text prompts: broad coverage of robot manipulation objects ──────
TEXT_PROMPTS = ["object"]

# ── Config ──────────────────────────────────────────────────────────
BOX_THRESHOLD = 0.2
MAX_CROPS_PER_VIDEO = 30
OCCLUSION_RATIO = 0.6     # mask_area / bbox_area below this → try completion
BBOX_EXPAND = 1.5         # expand bbox by this factor for completion hint
# ─────────────────────────────────────────────────────────────────────


CAMERA_KEYWORDS = ["head", "center", "high"]


def find_videos() -> list[Path]:
    """Non-wrist episode_000000.mp4 from head/center/high cameras only."""
    videos = []
    for p in sorted(VIDEO_ROOT.rglob("episode_000000.mp4")):
        s = str(p)
        if "wrist" in s or "chunk-000" not in s:
            continue
        if not any(kw in s for kw in CAMERA_KEYWORDS):
            continue
        videos.append(p)
    return videos


def get_first_frame_ffmpeg(video_path: Path) -> np.ndarray | None:
    """Extract first frame using ffmpeg (handles AV1)."""
    # Get dimensions
    probe = subprocess.run(
        [
            "ffprobe", "-v", "quiet", "-select_streams", "v:0",
            "-show_entries", "stream=width,height",
            "-of", "csv=p=0", str(video_path),
        ],
        capture_output=True, text=True, timeout=15,
    )
    if probe.returncode != 0:
        return None

    parts = probe.stdout.strip().split(",")
    if len(parts) < 2:
        return None
    w, h = int(parts[0]), int(parts[1])

    # Extract frame
    result = subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(video_path),
            "-vframes", "1", "-f", "image2pipe",
            "-pix_fmt", "rgb24", "-vcodec", "rawvideo", "-",
        ],
        capture_output=True, timeout=30,
    )
    if result.returncode != 0 or len(result.stdout) == 0:
        return None

    frame = np.frombuffer(result.stdout, dtype=np.uint8)
    expected = w * h * 3
    if len(frame) != expected:
        return None
    return frame.reshape(h, w, 3).copy()


def load_sam3():
    """Load SAM3 model and processor."""
    sys.path.insert(0, str(SAM3_REPO))
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    print(f"Loading SAM3 from {SAM3_CKPT}…", flush=True)

    model = build_sam3_image_model(
        checkpoint_path=str(SAM3_CKPT),
        bpe_path=str(BPE_PATH),
    ).cuda().eval()

    processor = Sam3Processor(model, confidence_threshold=0.2)
    return model, processor


def detect_objects(
    image: np.ndarray,
    model,
    processor,
) -> list[dict]:
    """
    Run SAM3 text-prompt detection.
    Returns list of {bbox, mask, score, prompt}.
    """
    h, w = image.shape[:2]
    pil_img = Image.fromarray(image)

    all_boxes = []
    all_masks = []
    all_scores = []
    all_prompts = []

    with torch.autocast("cuda", dtype=torch.bfloat16):
        inference_state = processor.set_image(pil_img)

        for prompt in TEXT_PROMPTS:
            try:
                output = processor.set_text_prompt(
                    state=inference_state, prompt=prompt
                )
            except Exception:
                continue

            boxes = output.get("boxes", [])
            masks = output.get("masks", [])
            scores = output.get("scores", [])

            if len(boxes) == 0:
                continue

            for i in range(len(boxes)):
                score = scores[i].item() if isinstance(scores[i], torch.Tensor) else scores[i]
                if score < BOX_THRESHOLD:
                    continue

                box = boxes[i]
                if isinstance(box, torch.Tensor):
                    box = box.tolist()

                mask = masks[i]
                if isinstance(mask, torch.Tensor):
                    mask = mask.cpu().numpy().squeeze()
                if mask.shape[:2] != (h, w):
                    continue

                all_boxes.append(box)
                all_masks.append(mask)
                all_scores.append(float(score))
                all_prompts.append(prompt)

    # NMS across all prompts
    if all_boxes:
        keep = _nms(all_boxes, all_scores, iou_threshold=0.5)
        all_boxes = [all_boxes[i] for i in keep]
        all_masks = [all_masks[i] for i in keep]
        all_scores = [all_scores[i] for i in keep]
        all_prompts = [all_prompts[i] for i in keep]

    results = []
    for box, mask, score, prompt in zip(
        all_boxes, all_masks, all_scores, all_prompts
    ):
        # Refine incomplete masks
        bbox_area = max((box[2] - box[0]) * (box[3] - box[1]), 1)
        ratio = mask.sum() / bbox_area
        if ratio < OCCLUSION_RATIO:
            refined = _refine_mask(image, box, mask, processor)
            if refined is not None:
                mask = refined

        results.append({
            "bbox": box,
            "mask": mask,
            "score": score,
            "prompt": prompt,
        })

    return results[:MAX_CROPS_PER_VIDEO]


def _refine_mask(
    image: np.ndarray,
    bbox: list,
    current_mask: np.ndarray,
    processor,
) -> np.ndarray | None:
    """
    Try to improve an incomplete mask by giving SAM3 an expanded bbox hint.
    Returns refined mask if it has more area, else None.
    """
    h, w = image.shape[:2]
    x1, y1, x2, y2 = bbox

    bw, bh = x2 - x1, y2 - y1
    cx = (x1 + x2) / 2 / w
    cy = (y1 + y2) / 2 / h
    cw = bw * BBOX_EXPAND / w
    ch = bh * BBOX_EXPAND / h

    box_norm = [
        max(0.0, min(1.0, cx)),
        max(0.0, min(1.0, cy)),
        max(0.01, min(1.0, cw)),
        max(0.01, min(1.0, ch)),
    ]

    pil_img = Image.fromarray(image)
    try:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            state = processor.set_image(pil_img)
            output = processor.add_geometric_prompt(box_norm, True, state)

        masks = output.get("masks", [])
        if len(masks) > 0:
            m = masks[0]
            if isinstance(m, torch.Tensor):
                m = m.cpu().numpy().squeeze()
            if m.shape[:2] == (h, w) and m.sum() > current_mask.sum():
                return m
    except Exception:
        pass
    return None


def _nms(boxes, scores, iou_threshold=0.5):
    """Simple NMS on xyxy boxes."""
    if not boxes:
        return []
    boxes_t = torch.tensor(boxes)
    scores_t = torch.tensor(scores)
    keep = torchvision.ops.nms(boxes_t, scores_t, iou_threshold)
    return keep.tolist()


# Predefined color palette for masks
COLORS = [
    (31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40),
    (148, 103, 189), (140, 86, 75), (227, 119, 194), (127, 127, 127),
    (188, 189, 34), (23, 190, 207), (31, 119, 180), (255, 127, 14),
    (44, 160, 44), (214, 39, 40), (148, 103, 189), (140, 86, 75),
    (227, 119, 194), (127, 127, 127), (188, 189, 34), (23, 190, 207),
    (174, 199, 232), (255, 187, 120), (152, 223, 138), (255, 152, 150),
    (197, 176, 213), (196, 156, 148), (247, 182, 210), (199, 199, 199),
    (219, 219, 141), (158, 218, 229), (100, 140, 200), (255, 100, 50),
]


def draw_annotated_frame(
    image_bgr: np.ndarray,
    detections: list[dict],
    alpha: float = 0.4,
) -> np.ndarray:
    """
    Draw all masks, bboxes, labels + scores on one image.
    Returns annotated BGR image (for cv2.imwrite).
    """
    import cv2

    h, w = image_bgr.shape[:2]
    overlay = image_bgr.copy()
    mask_canvas = np.zeros((h, w, 3), dtype=np.uint8)

    for i, det in enumerate(detections):
        color = COLORS[i % len(COLORS)]
        mask = det["mask"]

        # Colorize mask
        mask_canvas[mask] = color

        # Bbox
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)

        # Label
        prompt = det.get("prompt", "object")
        score = det["score"]
        label = f"{prompt} {score:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(
            overlay, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1,
        )
        cv2.putText(
            overlay, label, (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1,
        )

    # Blend mask overlay
    blended = cv2.addWeighted(overlay, 1.0, mask_canvas, alpha, 0)

    # Stamp summary at top-left
    n_obj = len(detections)
    cv2.putText(
        blended, f"Objects: {n_obj}", (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
    )

    return blended


def process_video(
    video_path: Path,
    model,
    processor,
) -> dict | None:
    """Detect objects in one video's first frame."""
    rel = video_path.relative_to(VIDEO_ROOT)
    parts = rel.parts
    dataset, camera = parts[0], parts[3]

    frame = get_first_frame_ffmpeg(video_path)
    if frame is None:
        return None

    detections = detect_objects(frame, model, processor)

    crop_dir = CROPS_DIR / dataset / camera
    crop_dir.mkdir(parents=True, exist_ok=True)

    # Save original frame
    orig_path = crop_dir / "frame.jpg"
    Image.fromarray(frame).save(str(orig_path))

    items = []
    for i, det in enumerate(detections):
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        h, w = frame.shape[:2]

        # Save full-frame mask (for object localization)
        full_mask = det["mask"]  # (h, w) boolean
        mask_path = crop_dir / f"seg_{i:02d}_mask.png"
        Image.fromarray((full_mask * 255).astype(np.uint8)).save(str(mask_path))

        # Crop with expanded bbox (for visual inspection of the object)
        pad_w = int((x2 - x1) * 0.05)
        pad_h = int((y2 - y1) * 0.05)
        cx1, cy1 = max(0, x1 - pad_w), max(0, y1 - pad_h)
        cx2, cy2 = min(w, x2 + pad_w), min(h, y2 + pad_h)

        crop = frame[cy1:cy2, cx1:cx2].copy()
        mask_roi = full_mask[cy1:cy2, cx1:cx2]
        crop_masked = crop.copy()
        crop_masked[~mask_roi] = 0
        crop_path = crop_dir / f"seg_{i:02d}_crop.jpg"
        Image.fromarray(crop_masked).save(str(crop_path))

        items.append({
            "dataset": dataset,
            "camera": camera,
            "video": str(rel),
            "seg_idx": i,
            "prompt": det["prompt"],
            "score": det["score"],
            "bbox": [cx1, cy1, cx2, cy2],
            "crop_path": str(crop_path.relative_to(BASE_DIR)),
            "mask_path": str(mask_path.relative_to(BASE_DIR)),
        })

    # Save annotated full-frame visualization
    if items:
        vis_path = crop_dir / "detection_vis.jpg"
        annotated = draw_annotated_frame(frame[..., ::-1], detections)  # RGB->BGR
        import cv2
        cv2.imwrite(str(vis_path), annotated)

    # Save per-video manifest
    manifest_path = MANIFEST_DIR / dataset / camera / "detections.jsonl"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    return {"video": str(rel), "detections": len(items)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    torch.cuda.set_device(args.gpu)

    # Need torchvision for NMS
    global torchvision
    import torchvision

    videos = find_videos()
    print(f"Found {len(videos)} videos", flush=True)
    if args.limit:
        videos = videos[:args.limit]

    model, processor = load_sam3()

    CROPS_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

    stats = {"videos": 0, "objects": 0, "errors": 0}
    t0 = time.time()

    for vpath in tqdm(videos, desc="Detecting"):
        try:
            result = process_video(vpath, model, processor)
            if result:
                stats["videos"] += 1
                stats["objects"] += result["detections"]
            else:
                stats["errors"] += 1
        except Exception as exc:
            stats["errors"] += 1
            tqdm.write(f"ERROR {vpath.relative_to(VIDEO_ROOT)}: {exc}")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s ({elapsed/60:.1f}m)")
    print(f"  Videos processed: {stats['videos']}")
    print(f"  Objects detected: {stats['objects']}")
    print(f"  Errors:           {stats['errors']}")
    print(f"  Crops → {CROPS_DIR}")
    print(f"  Manifests → {MANIFEST_DIR}")


if __name__ == "__main__":
    main()
