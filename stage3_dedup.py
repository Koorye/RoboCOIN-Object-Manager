#!/usr/bin/env python3
"""
Stage 3: Deduplication & Object Library

1. Compute CLIP visual embeddings + attribute text embeddings
2. DBSCAN clustering on fused embeddings
3. Build deduplicated object library

Same type + different color/material → different objects ✓
Red plastic bowl ≠ Blue plastic bowl ✓

Usage:
    conda activate robocoin-object
    python stage3_dedup.py
    python stage3_dedup.py --eps 0.30   # tighter matching
"""

import argparse
import json
import shutil
import time
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent
ATTR_CACHE = BASE_DIR / "objects" / "attributes.jsonl"
LIBRARY_DIR = BASE_DIR / "objects" / "library"
EMBEDDING_DIR = BASE_DIR / "objects" / "embeddings"

CLIP_MODEL_ID = "clip-ViT-B-16"
DBSCAN_EPS = 0.35
DBSCAN_MIN_SAMPLES = 2


def load_clip():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(CLIP_MODEL_ID)


def load_data() -> list[dict]:
    if not ATTR_CACHE.exists():
        raise FileNotFoundError(f"{ATTR_CACHE} not found. Run stage2 first.")
    data = []
    with open(ATTR_CACHE) as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    print(f"Loaded {len(data)} entries from {ATTR_CACHE}")
    return data


def attr_text(attrs: dict) -> str:
    inner = attrs.get("attributes", attrs)
    if not isinstance(inner, dict):
        return str(attrs)
    parts = []
    for k in ["category", "color", "material", "shape", "texture"]:
        v = inner.get(k, "")
        if v and v != "unknown":
            parts.append(str(v))
    return " ".join(parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eps", type=float, default=DBSCAN_EPS)
    args = parser.parse_args()

    data = load_data()
    clip = load_clip()

    # ── Compute embeddings ──
    print("Computing embeddings…", flush=True)
    visual_embs, text_embs, valid = [], [], []

    for item in data:
        cp = BASE_DIR / item.get("crop_path", "")
        if not cp.exists():
            continue
        try:
            img = Image.open(cp).convert("RGB")
            vemb = clip.encode(img, normalize_embeddings=True)
        except Exception:
            continue

        temb = clip.encode([attr_text(item)], normalize_embeddings=True)[0]
        visual_embs.append(vemb)
        text_embs.append(temb)
        valid.append(item)

    visual_embs = np.array(visual_embs, dtype=np.float32)
    text_embs = np.array(text_embs, dtype=np.float32)

    fused = np.concatenate([visual_embs, text_embs], axis=1)
    print(f"  Fused: {fused.shape}", flush=True)

    scaler = StandardScaler()
    fused_norm = scaler.fit_transform(fused)

    # ── Cluster ──
    print(f"DBSCAN(eps={args.eps}, min_samples={DBSCAN_MIN_SAMPLES})…", flush=True)
    labels = DBSCAN(
        eps=args.eps, min_samples=DBSCAN_MIN_SAMPLES, metric="cosine"
    ).fit_predict(fused_norm)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    print(f"  Clusters: {n_clusters}, Noise: {n_noise}", flush=True)

    # ── Group ──
    clusters = {}
    for idx, label in enumerate(labels):
        clusters.setdefault(int(label), []).append(valid[idx])

    # ── Build library ──
    LIBRARY_DIR.mkdir(parents=True, exist_ok=True)
    EMBEDDING_DIR.mkdir(parents=True, exist_ok=True)

    np.save(EMBEDDING_DIR / "visual.npy", visual_embs)
    np.save(EMBEDDING_DIR / "text.npy", text_embs)
    np.save(EMBEDDING_DIR / "labels.npy", labels)

    library = {}
    for label, items in clusters.items():
        if label == -1:
            for item in items:
                uid = f"obj_{abs(hash(item['crop_path'])) % 100000:05d}"
                _save_obj(uid, item, [item], library)
        else:
            uid = f"obj_{label:05d}"
            _save_obj(uid, items[0], items, library)

    # ── Save index ──
    index_path = LIBRARY_DIR / "index.json"
    with open(index_path, "w") as f:
        json.dump(library, f, indent=2, ensure_ascii=False)

    print(f"\nLibrary: {len(library)} unique objects → {index_path}")

    # ── Distribution ──
    cats = Counter()
    for obj in library.values():
        a = obj.get("attributes", {})
        cats[a.get("category", "?")] += 1
    print("Top categories:")
    for cat, cnt in cats.most_common(15):
        print(f"  {cat:20s} {cnt}")

    print(f"\nDone. Time: {time.time():.0f}s")


def _save_obj(uid: str, canonical: dict, items: list[dict], library: dict):
    obj_dir = LIBRARY_DIR / uid
    obj_dir.mkdir(parents=True, exist_ok=True)

    src = BASE_DIR / canonical["crop_path"]
    canon = obj_dir / "canonical.jpg"
    if src.exists() and not canon.exists():
        shutil.copy2(src, canon)

    # Majority vote on attributes
    merged = {}
    for key in ["category", "color", "material", "shape", "texture"]:
        vals = []
        for it in items:
            a = it.get("attributes", {})
            if isinstance(a, dict):
                if "attributes" in a:
                    a = a["attributes"]
                vals.append(str(a.get(key, "unknown")))
        if vals:
            merged[key] = Counter(vals).most_common(1)[0][0]

    entry = {
        "id": uid,
        "canonical_path": str(canon.relative_to(BASE_DIR)),
        "attributes": merged,
        "instance_count": len(items),
        "instances": [it["crop_path"] for it in items[:50]],
    }
    library[uid] = entry

    with open(obj_dir / "metadata.json", "w") as f:
        json.dump(entry, f, indent=2, ensure_ascii=False)
    with open(obj_dir / "instances.json", "w") as f:
        json.dump([it["crop_path"] for it in items], f, indent=2)


if __name__ == "__main__":
    main()
