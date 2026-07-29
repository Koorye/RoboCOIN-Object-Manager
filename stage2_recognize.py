#!/usr/bin/env python3
"""
Stage 2: VLM Recognition

Load crops from stage 1, run Qwen3-VL to extract structured attributes
for each object: {category, color, material, shape, texture}.

Output: objects/attributes.jsonl

Usage:
    conda activate robocoin-object
    python stage2_recognize.py                  # all crops
    python stage2_recognize.py --limit 50       # first 50
"""

import argparse
import json
import time
from pathlib import Path

import torch
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent
MANIFEST_DIR = BASE_DIR / "objects" / "manifests"
ATTR_CACHE = BASE_DIR / "objects" / "attributes.jsonl"

QWEN_MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"

PROMPT = """Describe this object precisely. Output ONLY a JSON:
{
  "category": "object category (e.g. bowl, cup, toy, block, fruit, bottle, plate, towel, book, box, pen, remote, phone, basket, sponge, brush, cable, ball, doll, apple, banana, lemon, orange, pear, peach, bread, can, bag, scissors, tape, lid, cap, tray, tool, container, clothes, electronic, utensil, vegetable, cloth)",
  "color": "primary color(s) (e.g. red, blue, green, yellow, white, black, brown, orange, purple, pink, gray, silver, gold, transparent, beige, multicolored)",
  "material": "material (e.g. plastic, metal, ceramic, wood, glass, rubber, fabric, paper, foam, silicone, cardboard)",
  "shape": "shape (e.g. round, square, rectangular, cylindrical, spherical, irregular, flat, curved, cuboid)",
  "texture": "surface (e.g. smooth, rough, glossy, matte, textured, ribbed, bumpy, patterned)"
}
Output ONLY the JSON, no other text."""


def load_qwen_vl():
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading Qwen3-VL ({QWEN_MODEL_ID}) on {device}…", flush=True)

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        QWEN_MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(QWEN_MODEL_ID)
    model.eval()
    return model, processor, device


def recognize(crop_path: Path, model, processor, device: str) -> dict:
    image = Image.open(crop_path).convert("RGB")

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": PROMPT},
        ],
    }]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs, max_new_tokens=200, temperature=0.1, do_sample=False
        )

    input_len = inputs.input_ids.shape[1]
    output = processor.decode(
        generated_ids[0, input_len:], skip_special_tokens=True
    ).strip()

    try:
        if "```json" in output:
            output = output.split("```json")[1].split("```")[0]
        elif "```" in output:
            output = output.split("```")[1].split("```")[0]
        return json.loads(output)
    except (json.JSONDecodeError, IndexError):
        return {"raw_output": output, "error": "json_parse_failed"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    manifest_path = MANIFEST_DIR / "segments.json"
    if not manifest_path.exists():
        print(f"ERROR: {manifest_path} not found. Run stage1 first.")
        return

    with open(manifest_path) as f:
        crops = json.load(f)

    print(f"Loaded {len(crops)} crops from manifest")
    if args.limit:
        crops = crops[:args.limit]
        print(f"Limited to {args.limit}")

    # Resume: check existing attributes
    existing_paths = set()
    if ATTR_CACHE.exists():
        with open(ATTR_CACHE) as f:
            for line in f:
                try:
                    existing_paths.add(json.loads(line)["crop_path"])
                except Exception:
                    pass
        print(f"Found {len(existing_paths)} already processed (will skip)")

    model, processor, device = load_qwen_vl()

    results = []
    t0 = time.time()
    processed = 0

    with open(ATTR_CACHE, "a") as fout:
        for i, meta in enumerate(crops, 1):
            if meta["crop_path"] in existing_paths:
                continue

            crop_path = BASE_DIR / meta["crop_path"]
            if not crop_path.exists():
                continue

            try:
                attrs = recognize(crop_path, model, processor, device)
            except Exception as exc:
                attrs = {"error": str(exc)}

            entry = {**meta, "attributes": attrs}
            fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
            fout.flush()
            results.append(entry)
            processed += 1

            if i % 10 == 0 or i == len(crops):
                elapsed = time.time() - t0
                print(
                    f"\r[{i:5d}/{len(crops)}] new={processed} "
                    f"({i / elapsed:.1f}/s)", end="", flush=True,
                )

    print(f"\nDone: {processed} new, {len(existing_paths)} skipped")
    print(f"Output: {ATTR_CACHE}")


if __name__ == "__main__":
    main()
