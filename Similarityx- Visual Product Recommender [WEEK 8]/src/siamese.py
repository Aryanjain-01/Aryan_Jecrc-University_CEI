"""
siamese.py — Iteration 2 metric-learning model.

Two pieces:
  * build_projection_head()        — the shared encoder (Step 1): frozen 2048-d
                                     features -> L2-normalized 128-d embedding.
  * build_siamese_triplet_model()  — the training harness (Step 2): three inputs
                                     (anchor, positive, negative), the SAME head
                                     applied to each, outputting the three stacked
                                     embeddings for the triplet loss (Step 4).

The ResNet50 backbone stays FROZEN; we train only the head, on the precomputed
2048-d baseline embeddings -> fast on CPU.
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers, Model

EMBED_DIM = 2048        # ResNet50 GAP output dimension (input to the head)
PROJECTION_DIM = 128    # learned metric-space dimension (like FaceNet)


def build_projection_head(input_dim: int = EMBED_DIM,
                          projection_dim: int = PROJECTION_DIM) -> Model:
    """Shared encoder: frozen 2048-d features -> L2-normalized 128-d embedding.

    2048 -> 512 -> 256 -> 128, then L2-normalize onto the unit hypersphere.
    ReLU + BatchNorm + Dropout regularize (we have only ~4k items); the final
    Dense has no activation so we don't clip before normalizing.
    """
    inputs = layers.Input(shape=(input_dim,), name="frozen_embedding")
    x = layers.Dense(512, activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(projection_dim, activation=None)(x)
    outputs = layers.UnitNormalization(name="l2_normalize")(x)   # unit hypersphere
    return Model(inputs, outputs, name="projection_head")


def build_siamese_triplet_model(head: Model | None = None,
                                input_dim: int = EMBED_DIM):
    """Triplet Siamese wrapper. Returns (siamese_model, shared_head).

    Three inputs (anchor, positive, negative), each a 2048-d frozen embedding.
    The SINGLE `head` object is applied to all three -> weights are SHARED (this
    is the crux of a Siamese net). Output is the three embeddings stacked into
    shape (batch, 3, 128) so Step 4's triplet loss can slice them apart.
    """
    if head is None:
        head = build_projection_head(input_dim)

    anchor_in = layers.Input(shape=(input_dim,), name="anchor")
    positive_in = layers.Input(shape=(input_dim,), name="positive")
    negative_in = layers.Input(shape=(input_dim,), name="negative")

    # SAME object called 3x -> shared weights (do NOT build three heads).
    emb_a = head(anchor_in)
    emb_p = head(positive_in)
    emb_n = head(negative_in)

    stacked = layers.Lambda(
        lambda t: tf.stack(t, axis=1),
        output_shape=(3, PROJECTION_DIM),
        name="triplet_embeddings",
    )([emb_a, emb_p, emb_n])                       # (batch, 3, 128)

    model = Model(inputs=[anchor_in, positive_in, negative_in],
                  outputs=stacked, name="siamese_triplet")
    return model, head


if __name__ == "__main__":
    import numpy as np

    # --- Step 1 head check ---
    head = build_projection_head()
    out = head(np.random.rand(4, EMBED_DIM).astype("float32"), training=False).numpy()
    print("head output :", out.shape, "| L2 norms:", np.round(np.linalg.norm(out, axis=1), 3))

    # --- Step 2 triplet model check ---
    siamese, shared = build_siamese_triplet_model(head)
    a = np.random.rand(4, EMBED_DIM).astype("float32")
    p = np.random.rand(4, EMBED_DIM).astype("float32")
    n = np.random.rand(4, EMBED_DIM).astype("float32")
    triplet_out = siamese([a, p, n], training=False).numpy()
    print("triplet out :", triplet_out.shape, "(expect (4, 3, 128))")

    # Proof of weight sharing: the siamese model has the SAME number of trainable
    # weight tensors as ONE head (because it reuses that one head, not 3 copies).
    print("head trainable tensors    :", len(head.trainable_weights))
    print("siamese trainable tensors :", len(siamese.trainable_weights),
          "-> shared!" if len(siamese.trainable_weights) == len(head.trainable_weights)
          else "-> NOT shared (bug)")
