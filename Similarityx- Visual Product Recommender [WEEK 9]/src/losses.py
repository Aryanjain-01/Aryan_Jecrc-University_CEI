"""
losses.py — Triplet loss (and a companion metric) for the Siamese network.

Operates on the Siamese model's stacked output of shape (batch, 3, 128):
    y_pred[:, 0, :] = anchor embedding
    y_pred[:, 1, :] = positive embedding
    y_pred[:, 2, :] = negative embedding

Triplet loss:
    L = mean( max(0,  d(A,P) - d(A,N) + margin) )
where d is SQUARED Euclidean distance (smooth gradients; standard in FaceNet).
Because embeddings are L2-normalized, squared distance is monotonic with cosine.
y_true is unused — all supervision lives in the geometry of y_pred.
"""

from __future__ import annotations

import tensorflow as tf

DEFAULT_MARGIN = 0.2


def _split(y_pred):
    """Split (batch, 3, dim) -> anchor, positive, negative."""
    return y_pred[:, 0, :], y_pred[:, 1, :], y_pred[:, 2, :]


def triplet_loss(margin: float = DEFAULT_MARGIN):
    """Return a Keras-compatible triplet loss with the given margin."""
    def loss(y_true, y_pred):
        anchor, positive, negative = _split(y_pred)
        d_ap = tf.reduce_sum(tf.square(anchor - positive), axis=1)   # (batch,)
        d_an = tf.reduce_sum(tf.square(anchor - negative), axis=1)   # (batch,)
        return tf.reduce_mean(tf.maximum(0.0, d_ap - d_an + margin))
    return loss


def triplet_accuracy(y_true, y_pred):
    """Fraction of triplets already correctly ordered: d(A,N) > d(A,P).

    A useful training signal: it should climb toward ~1.0 as the space learns to
    place the negative farther than the positive. (Margin-free — pure ordering.)
    """
    anchor, positive, negative = _split(y_pred)
    d_ap = tf.reduce_sum(tf.square(anchor - positive), axis=1)
    d_an = tf.reduce_sum(tf.square(anchor - negative), axis=1)
    return tf.reduce_mean(tf.cast(d_an > d_ap, tf.float32))


if __name__ == "__main__":
    import numpy as np

    def np_triplet_loss(y, margin=DEFAULT_MARGIN):
        a, p, n = y[:, 0, :], y[:, 1, :], y[:, 2, :]
        d_ap = np.sum((a - p) ** 2, axis=1)
        d_an = np.sum((a - n) ** 2, axis=1)
        return np.mean(np.maximum(0.0, d_ap - d_an + margin))

    dim = 4
    # Case 1: well-separated (P near A, N far) -> loss should be 0.
    good = np.array([[[1, 0, 0, 0], [0.99, 0.1, 0, 0], [-1, 0, 0, 0]]], dtype="float32")
    # Case 2: violating (N closer to A than P) -> loss should be > 0.
    bad = np.array([[[1, 0, 0, 0], [-1, 0, 0, 0], [0.99, 0.1, 0, 0]]], dtype="float32")
    print("well-separated triplet loss :", round(float(np_triplet_loss(good)), 4), "(expect 0.0)")
    print("violating triplet loss      :", round(float(np_triplet_loss(bad)), 4), "(expect > 0)")
