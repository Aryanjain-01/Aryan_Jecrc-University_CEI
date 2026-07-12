"""
triplets.py — Sample (anchor, positive, negative) triplets for training.

Supervision comes from the `category` label:
    * positive = a DIFFERENT item of the SAME category as the anchor
    * negative = an item of a DIFFERENT category

We operate on the precomputed 2048-d embeddings (embeddings.pkl), so a "triplet"
is just three saved vectors selected by row index. Triplets are resampled every
epoch (variety = regularization; the model can't memorize a fixed list).
"""

from __future__ import annotations

import numpy as np


class TripletSampler:
    """Draws random (anchor, positive, negative) triplets by category."""

    def __init__(self, embeddings=None, categories=None, seed: int = 42):
        if embeddings is None:
            # Lazy import so this module doesn't pull in TensorFlow unless needed.
            from src.embeddings import load_embeddings
            data = load_embeddings()
            embeddings = data["embeddings"]
            categories = data["metadata"]["category"].to_numpy()

        self.embeddings = np.asarray(embeddings, dtype="float32")
        self.categories = np.asarray(categories)
        self.rng = np.random.default_rng(seed)
        self.all_idx = np.arange(len(self.categories))
        # Precompute the row indices belonging to each category for fast sampling.
        self.by_cat = {c: np.where(self.categories == c)[0]
                       for c in np.unique(self.categories)}

    def sample_triplet_indices(self, n: int):
        """Return (anchor_idx, positive_idx, negative_idx) arrays of length n."""
        anchors = self.rng.choice(self.all_idx, size=n)
        positives = np.empty(n, dtype=int)
        negatives = np.empty(n, dtype=int)

        for i, a in enumerate(anchors):
            cat = self.categories[a]
            same = self.by_cat[cat]

            # Positive: same category, but NOT the anchor itself.
            p = a
            while p == a:
                p = self.rng.choice(same)
            positives[i] = p

            # Negative: any item from a DIFFERENT category.
            neg = a
            while self.categories[neg] == cat:
                neg = self.rng.choice(self.all_idx)
            negatives[i] = neg

        return anchors, positives, negatives

    def sample_batch(self, n: int):
        """Return ([A, P, N] embedding arrays, dummy_y) for model.fit()."""
        a, p, neg = self.sample_triplet_indices(n)
        inputs = [self.embeddings[a], self.embeddings[p], self.embeddings[neg]]
        # Triplet loss uses only y_pred (the embeddings); y_true is a placeholder.
        y = np.zeros((n,), dtype="float32")
        return inputs, y

    def generator(self, batch_size: int = 64, batches_per_epoch: int = 100):
        """Infinite generator of fresh triplet batches (for model.fit)."""
        while True:
            for _ in range(batches_per_epoch):
                yield self.sample_batch(batch_size)


if __name__ == "__main__":
    sampler = TripletSampler()
    a, p, n = sampler.sample_triplet_indices(2000)
    cats = sampler.categories

    pos_ok = np.mean(cats[a] == cats[p])          # positives should match anchor cat
    neg_ok = np.mean(cats[a] != cats[n])          # negatives should differ
    self_ok = np.mean(a == p)                     # anchor should never equal positive

    print(f"triplets sampled       : {len(a)}")
    print(f"positive same-category : {pos_ok*100:.1f}%  (expect 100.0%)")
    print(f"negative diff-category : {neg_ok*100:.1f}%  (expect 100.0%)")
    print(f"anchor == positive     : {self_ok*100:.1f}%  (expect 0.0%)")

    (A, P, N), y = sampler.sample_batch(64)
    print(f"batch shapes           : A{A.shape} P{P.shape} N{N.shape} y{y.shape}")
