"""
improved_embeddings.py — Re-embed the catalog through the trained head (Step 6).

Baseline embeddings live in the 2048-d ImageNet space. Here we map every item
into the new fashion-tuned 128-d space learned in Step 5, and save it as a fresh
index (improved_embeddings.pkl), aligned exactly like the baseline one.

Because the ResNet50 backbone is frozen, its 2048-d outputs never changed — so we
just push the PRECOMPUTED baseline vectors through the head (no images touched).

Also exposes embed_image_improved() for query time (Step 7 / the app), which runs
the FULL composition: raw image -> frozen ResNet50 -> trained head -> 128-d.

Run:
    python -m src.improved_embeddings
"""

from __future__ import annotations

import pickle

import numpy as np
from tensorflow.keras.models import load_model

from src.config import EMBEDDINGS_DIR, ARTIFACTS_DIR
from src.embeddings import load_embeddings

HEAD_PATH = ARTIFACTS_DIR / "models" / "projection_head.keras"
IMPROVED_PKL = EMBEDDINGS_DIR / "improved_embeddings.pkl"

_head = None  # lazy singleton


def get_head():
    """Load (once) the trained projection head."""
    global _head
    if _head is None:
        _head = load_model(HEAD_PATH)
    return _head


def generate_improved_embeddings(batch_size: int = 256) -> dict:
    """Map baseline 2048-d catalog vectors -> improved 128-d, save aligned bundle."""
    data = load_embeddings()                       # baseline 2048-d + metadata
    base = data["embeddings"]                      # (N, 2048)
    improved = get_head().predict(base, batch_size=batch_size, verbose=0)
    improved = improved.astype("float32")          # (N, 128), already L2-normalized

    payload = {
        "embeddings": improved,                    # (N, 128)
        "ids": data["ids"],
        "image_paths": data["image_paths"],
        "metadata": data["metadata"],              # same alignment as baseline
    }
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    with open(IMPROVED_PKL, "wb") as f:
        pickle.dump(payload, f)
    print(f"[ok] improved embeddings saved -> {IMPROVED_PKL}")
    return payload


def embed_image_improved(source) -> np.ndarray:
    """Query-time: raw image -> frozen ResNet50 -> trained head -> (128,) vector."""
    from src.feature_extractor import get_embedding      # lazy: avoids circular import
    base = get_embedding(source).reshape(1, -1)          # (1, 2048)
    return get_head().predict(base, verbose=0)[0]        # (128,)


if __name__ == "__main__":
    payload = generate_improved_embeddings()
    emb = payload["embeddings"]
    print("improved matrix :", emb.shape, emb.dtype)     # expect (3964, 128) float32
    print("L2 norms (~1)   :", np.round(np.linalg.norm(emb[:4], axis=1), 4))
