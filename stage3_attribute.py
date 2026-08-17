#!/usr/bin/env python3
"""
Stage 3: Attribute Annotation with Qwen3-VL

For each crop from stage1/2, run Qwen3-VL to extract structured attributes:
{category, color, material, shape, texture}

Output: objects/attributes.jsonl

Usage:
    python stage3_attribute.py
    python stage3_attribute.py --limit 100
"""

import argparse
import json
import time
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent
CROPS_DIR = BASE_DIR / "objects" / "crops"
ATTR_CACHE = BASE_DIR / "objects" / "attributes.jsonl"

QWEN_MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"

PROMPT = """Describe this object precisely. Output ONLY a JSON:
{
  "category": "object category (e.g. bowl, cup, toy, block, fruit, bottle, plate, towel, book, box, pen, remote, phone, basket, sponge, brush, cable, ball, doll, apple, banana, lemon, orange, pear, peach, bread, can, bag, scissors, tape, lid, cap, tray, tool, container, clothes, electronic, utensil, vegetable, cloth)",
  "color": "primary color(s) (e.g. red, blue, green, yellow, white, black, brown, orange, purple, pink, gray, silver, gold, transparent, beige, multicolored)",
  "material": "material (e.g. plastic, metal, ceramic, wood, glass, rubber, fabric, paper, foam, silicone, cardboard)",
  "shape": "shape (e.g. round, square, rectangular, cylindrical, spherical, irregular, flat, curved, cuboid)",
  "texture": "surface (e.g. smooth, rough, glossy, matte, textured, ribbed, bumpy, patterned)"
}
Output ONLY the JSON, no other text."""


def load_qwen():
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {QWEN_MODEL_ID}…", flush=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        QWEN_MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(QWEN_MODEL_ID)
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
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(device)

    with torch.no_grad():
        gen = model.generate(**inputs, max_new_tokens=200, temperature=0.1, do_sample=False)
    out = processor.decode(gen[0, inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

    try:
        if "```json" in out:
            out = out.split("```json")[1].split("```")[0]
        elif "```" in out:
            out = out.split("```")[1].split("```")[0]
        return json.loads(out)
    except (json.JSONDecodeError, IndexError):
        return {"raw_output": out, "error": "json_parse_failed"}


def find_crops() -> list[Path]:
    return sorted(CROPS_DIR.rglob("*_crop.jpg"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    crops = find_crops()
    print(f"Found {len(crops)} crops", flush=True)
    if args.limit:
        crops = crops[: args.limit]

    # Resume
    existing = set()
    if ATTR_CACHE.exists():
        with open(ATTR_CACHE) as f:
            for line in f:
                try:
                    existing.add(json.loads(line)["crop_path"])
                except Exception:
                    pass
        print(f"  {len(existing)} already processed (will skip)")

    model, processor, device = load_qwen()

    processed = 0
    t0 = time.time()
    with open(ATTR_CACHE, "a") as fout:
        for cp in tqdm(crops, desc="Recognizing"):
            rel = str(cp.relative_to(BASE_DIR))
            if rel in existing:
                continue
            try:
                attrs = recognize(cp, model, processor, device)
            except Exception as e:
                attrs = {"error": str(e)}
            entry = {"crop_path": rel, "attributes": attrs}
            fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
            fout.flush()
            processed += 1

    print(f"Done: {processed} new, {len(existing)} skipped ({time.time() - t0:.0f}s)")
    print(f"Output: {ATTR_CACHE}")


if __name__ == "__main__":
    main()
