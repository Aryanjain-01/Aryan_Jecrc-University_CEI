"""
recommender.py — Cosine Top-K retrieval over the precomputed catalog embeddings.

Loads the pickled index once, then answers queries in milliseconds:
    query image -> embedding -> cosine similarity vs all catalog vectors
                -> sort descending -> return Top-K matches (+ scores + metadata)

Speed trick: we L2-NORMALIZE the catalog embeddings once at load time. For unit
vectors, cosine similarity is just the dot product, so scoring the whole catalog
becomes a single matrix-vector multiply (self._normed @ q_norm).
"""

from __future__ import annotations

import time

import numpy as np

from src.embeddings import load_embeddings
from src.feature_extractor import get_embedding

_EPS = 1e-8


def _l2_normalize(x: np.ndarray, axis: int) -> np.ndarray:
    """Divide vectors by their L2 norm so cosine == dot product."""
    norms = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.clip(norms, _EPS, None)


class Recommender:
    """Holds the catalog index and serves Top-K visual recommendations."""

    def __init__(self):
        data = load_embeddings()
        self.embeddings = data["embeddings"]          # (N, 2048) raw
        self.ids = data["ids"]                        # (N,)
        self.image_paths = data["image_paths"]        # length N
        self.metadata = data["metadata"]              # N rows, aligned
        # Pre-normalize once -> cosine similarity becomes a dot product.
        self._normed = _l2_normalize(self.embeddings, axis=1).astype("float32")

    def recommend(self, source, k: int = 5, exclude_id: int | None = None):
        """Return (results, elapsed_ms).

        `source`     : image path / PIL image / file-like (Streamlit upload).
        `exclude_id` : if the query IS a catalog item, pass its id to drop the
                       self-match. If None, we instead drop any near-1.0 hit
                       (handles an uploaded image identical to a catalog image).
        `results`    : list of dicts with score + id + image_path + metadata,
                       ranked most-similar first.
        """
        t0 = time.time()
        q = get_embedding(source).astype("float32")
        q = _l2_normalize(q, axis=0)
        sims = self._normed @ q                       # (N,) cosine similarities
        order = np.argsort(-sims)                     # indices, highest first

        results = []
        for idx in order:
            score = float(sims[idx])
            row_id = int(self.ids[idx])
            if exclude_id is not None and row_id == exclude_id:
                continue
            if exclude_id is None and score >= 0.9999:  # self / exact duplicate
                continue
            meta = self.metadata.iloc[idx].to_dict()
            results.append({"score": score, "id": row_id,
                            "image_path": self.image_paths[idx], **meta})
            if len(results) == k:
                break

        elapsed_ms = (time.time() - t0) * 1000.0
        return results, elapsed_ms


if __name__ == "__main__":
    import random
    import pandas as pd
    from src.config import SUBSET_CSV, PROJECT_ROOT

    df = pd.read_csv(SUBSET_CSV)
    row = df.iloc[random.randint(0, len(df) - 1)]        # pick a random query
    query_path = PROJECT_ROOT / row["image_path"]
    print(f"QUERY: {row['productDisplayName']}  [{row['category']}]  (id={row['id']})\n")

    rec = Recommender()
    results, ms = rec.recommend(query_path, k=5, exclude_id=int(row["id"]))

    print(f"Top-5 in {ms:.1f} ms:")
    for i, r in enumerate(results, 1):
        print(f"  {i}. {r['score']:.3f}  {r['productDisplayName'][:45]:45}  [{r['category']}]")

    same = sum(r["category"] == row["category"] for r in results)
    print(f"\nsame-category matches: {same}/5  (sanity check)")
