"""
embeddings.py — Generate and store embeddings for the entire subset.

This is the OFFLINE step: we run the frozen ResNet50 encoder over all ~4,000
catalog images once, then pickle the result so every future query is cheap.

We save a single dictionary that keeps four things ALIGNED by row index:
    embeddings[i]  <->  ids[i]  <->  image_paths[i]  <->  metadata.iloc[i]
If that alignment ever breaks, recommendations point at the wrong products, so
we build all four from the same DataFrame in the same order and never reshuffle.

Run:
    python -m src.embeddings
"""

from __future__ import annotations

import pickle
import time

import numpy as np
import pandas as pd

from src.config import SUBSET_CSV, EMBEDDINGS_DIR, PROJECT_ROOT
from src.feature_extractor import get_model
from src.preprocessing import preprocess_batch

EMBEDDINGS_PKL = EMBEDDINGS_DIR / "embeddings.pkl"

# Metadata columns we want available at recommendation time (for display).
META_COLUMNS = [
    "id", "productDisplayName", "category",
    "gender", "baseColour", "season", "year", "usage", "image_path",
]


def generate_embeddings(batch_size: int = 32) -> dict:
    """Embed every subset image and pickle the aligned bundle. Returns it too."""
    df = pd.read_csv(SUBSET_CSV).reset_index(drop=True)
    model = get_model()
    n = len(df)
    print(f"Embedding {n} images (batch_size={batch_size}) ...")

    chunks = []
    t0 = time.time()
    for start in range(0, n, batch_size):
        rows = df.iloc[start:start + batch_size]
        paths = [PROJECT_ROOT / p for p in rows["image_path"]]
        batch = preprocess_batch(paths)                       # (b, 224, 224, 3)
        chunks.append(model.predict(batch, verbose=0))        # (b, 2048)
        done = min(start + batch_size, n)
        print(f"  {done}/{n}  ({done * 100 // n}%)", end="\r", flush=True)

    embeddings = np.vstack(chunks).astype("float32")          # (n, 2048)
    elapsed = time.time() - t0
    print(f"\nEmbedded {n} images in {elapsed:.1f}s "
          f"({elapsed / n * 1000:.0f} ms/image). Matrix: {embeddings.shape}")

    # Sanity: alignment invariant — one embedding row per catalog row.
    assert embeddings.shape[0] == n, "embedding/row count mismatch!"

    meta_cols = [c for c in META_COLUMNS if c in df.columns]
    payload = {
        "embeddings": embeddings,                 # (n, 2048) float32
        "ids": df["id"].to_numpy(),               # (n,)
        "image_paths": df["image_path"].tolist(),  # length n
        "metadata": df[meta_cols].reset_index(drop=True),  # n rows, aligned
    }

    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    with open(EMBEDDINGS_PKL, "wb") as f:
        pickle.dump(payload, f)
    size_mb = EMBEDDINGS_PKL.stat().st_size / 1e6
    print(f"[ok] saved {EMBEDDINGS_PKL}  ({size_mb:.1f} MB)")
    return payload


def load_embeddings() -> dict:
    """Load the pickled bundle produced by generate_embeddings()."""
    with open(EMBEDDINGS_PKL, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    payload = generate_embeddings()
    print("\n--- summary ---")
    print("embeddings :", payload["embeddings"].shape, payload["embeddings"].dtype)
    print("ids        :", payload["ids"].shape)
    print("metadata   :", payload["metadata"].shape, "->", list(payload["metadata"].columns))
    print("per-category counts:\n", payload["metadata"]["category"].value_counts())
