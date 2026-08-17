#!/usr/bin/env python3
"""
Stage 4: Deduplication & Object Library

1. Compute CLIP visual + attribute-text embeddings
2. DBSCAN clustering on fused embeddings
3. Build deduplicated object library

Same type + different color/material → different objects
(e.g. red plastic bowl ≠ blue plastic bowl)

Usage:
    python stage4_dedup.py
    python stage4_dedup.py --eps 0.30  # tighter matching
"""

import argparse
import json
import shutil
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import clip
from PIL import Image
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent
ATTR_CACHE = BASE_DIR / "objects" / "attributes.jsonl"
LIBRARY_DIR = BASE_DIR / "objects" / "library"
EMBEDDING_DIR = BASE_DIR / "objects" / "embeddings"

CLIP_MODEL_ID = "ViT-B/16"
DBSCAN_EPS = 0.35
DBSCAN_MIN_SAMPLES = 2


def load_clip(download_root: str = None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(CLIP_MODEL_ID, device=device, download_root=download_root)
    return model, preprocess, device


def load_data():
    if not ATTR_CACHE.exists():
        raise FileNotFoundError(f"{ATTR_CACHE} not found. Run stage3 first.")
    data = []
    with open(ATTR_CACHE) as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    print(f"Loaded {len(data)} entries")
    return data


def attr_text(item: dict) -> str:
    a = item.get("attributes", {})
    if not isinstance(a, dict):
        return str(a)
    parts = []
    for k in ["category", "color", "material", "shape", "texture"]:
        v = a.get(k, "")
        if v and v != "unknown":
            parts.append(str(v))
    return " ".join(parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eps", type=float, default=DBSCAN_EPS)
    parser.add_argument("--clip-dir", type=str, default=None,
                        help="Directory containing CLIP model checkpoint (e.g. ViT-B-16.pt)")
    args = parser.parse_args()

    data = load_data()
    model, preprocess, device = load_clip(download_root=args.clip_dir)

    # Embeddings
    print("Computing embeddings…", flush=True)
    visual, text, valid = [], [], []
    for item in data:
        cp = BASE_DIR / item.get("crop_path", "")
        if not cp.exists():
            continue
        try:
            image = preprocess(Image.open(cp).convert("RGB")).unsqueeze(0).to(device)
            with torch.no_grad():
                v = model.encode_image(image)
                v = (v / v.norm(dim=-1, keepdim=True)).cpu().numpy()[0]

            tokenized = clip.tokenize([attr_text(item)]).to(device)
            with torch.no_grad():
                t = model.encode_text(tokenized)
                t = (t / t.norm(dim=-1, keepdim=True)).cpu().numpy()[0]
        except Exception:
            continue
        visual.append(v)
        text.append(t)
        valid.append(item)

    visual = np.array(visual, dtype=np.float32)
    text = np.array(text, dtype=np.float32)
    fused = np.concatenate([visual, text], axis=1)
    print(f"  Fused: {fused.shape}", flush=True)

    fused_norm = StandardScaler().fit_transform(fused)

    # Cluster
    print(f"DBSCAN(eps={args.eps}, min_samples={DBSCAN_MIN_SAMPLES})…", flush=True)
    labels = DBSCAN(
        eps=args.eps, min_samples=DBSCAN_MIN_SAMPLES, metric="cosine"
    ).fit_predict(fused_norm)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    print(f"  Clusters: {n_clusters}, Noise: {n_noise}", flush=True)

    # Group
    clusters = {}
    for idx, label in enumerate(labels):
        clusters.setdefault(int(label), []).append(valid[idx])

    # Build library
    LIBRARY_DIR.mkdir(parents=True, exist_ok=True)
    EMBEDDING_DIR.mkdir(parents=True, exist_ok=True)
    np.save(EMBEDDING_DIR / "visual.npy", visual)
    np.save(EMBEDDING_DIR / "text.npy", text)
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

    # Index
    idx_path = LIBRARY_DIR / "index.json"
    with open(idx_path, "w") as f:
        json.dump(library, f, indent=2, ensure_ascii=False)

    print(f"\nLibrary: {len(library)} unique objects → {idx_path}")

    cats = Counter()
    for obj in library.values():
        a = obj.get("attributes", {})
        cats[a.get("category", "?")] += 1
    print("Top categories:")
    for cat, cnt in cats.most_common(15):
        print(f"  {cat:20s} {cnt}")


def _save_obj(uid: str, canonical: dict, items: list[dict], library: dict):
    obj_dir = LIBRARY_DIR / uid
    obj_dir.mkdir(parents=True, exist_ok=True)

    src = BASE_DIR / canonical["crop_path"]
    canon = obj_dir / "canonical.jpg"
    if src.exists() and not canon.exists():
        shutil.copy2(src, canon)

    # Majority-vote attributes
    merged = {}
    for key in ["category", "color", "material", "shape", "texture"]:
        vals = []
        for it in items:
            a = it.get("attributes", {})
            if isinstance(a, dict):
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
