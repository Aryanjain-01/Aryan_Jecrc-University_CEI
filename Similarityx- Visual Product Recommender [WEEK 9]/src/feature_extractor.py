"""
feature_extractor.py — Headless, frozen ResNet50 that turns an image into a
2048-dimensional embedding.

This is the "encoder" from Step 4:
    image -> preprocess -> ResNet50(include_top=False, pooling="avg", frozen) -> 2048-d vector

Design notes:
  * The model is built ONCE and cached (a lazy singleton). Loading ImageNet
    weights is slow, so we never want to rebuild it per call.
  * `include_top=False` removes the 1000-class classification head.
  * `pooling="avg"` bolts Global Average Pooling onto the 7x7x2048 feature map,
    so the model outputs a clean (batch, 2048) tensor directly.
  * `trainable = False` freezes every weight: this is FEATURE EXTRACTION, not
    training. In Iteration 2 we will unfreeze the top layers (fine-tuning).
"""

from __future__ import annotations

import numpy as np
from tensorflow.keras.applications import ResNet50

from src.config import IMAGE_SIZE
from src.preprocessing import preprocess_image, preprocess_batch

# Lazy singleton — built on first use, reused forever after.
_model = None


def get_model():
    """Build (once) and return the frozen, headless ResNet50 encoder."""
    global _model
    if _model is None:
        _model = ResNet50(
            include_top=False,            # drop the 1000-class ImageNet head
            weights="imagenet",           # reuse pretrained features (transfer learning)
            pooling="avg",                # Global Average Pooling -> (None, 2048)
            input_shape=(*IMAGE_SIZE, 3),  # (224, 224, 3)
        )
        _model.trainable = False          # freeze all weights: pure feature extraction
    return _model


def get_embedding(source) -> np.ndarray:
    """Embed a single image (path / PIL / file-like) -> (2048,) float32 vector."""
    x = preprocess_image(source)                     # (224, 224, 3)
    x = np.expand_dims(x, axis=0)                    # (1, 224, 224, 3) — add batch dim
    emb = get_model().predict(x, verbose=0)          # (1, 2048)
    return emb[0]                                    # (2048,)


def get_embeddings(sources, batch_size: int = 32) -> np.ndarray:
    """Embed many images -> (N, 2048) float32 array (batched forward pass)."""
    batch = preprocess_batch(sources)                # (N, 224, 224, 3)
    return get_model().predict(batch, batch_size=batch_size, verbose=0)  # (N, 2048)


if __name__ == "__main__":
    # Smoke test: embed the first subset image and report the vector.
    import pandas as pd
    from src.config import SUBSET_CSV, PROJECT_ROOT

    df = pd.read_csv(SUBSET_CSV)
    sample_path = PROJECT_ROOT / df.iloc[0]["image_path"]

    model = get_model()
    print(f"model output shape : {model.output_shape}")   # expect (None, 2048)
    print(f"trainable params   : {sum(w.shape.num_elements() for w in model.trainable_weights)}")
    print("  (expect 0 — the encoder is frozen)")

    emb = get_embedding(sample_path)
    print(f"\nsample : {sample_path.name}")
    print(f"shape  : {emb.shape}")        # expect (2048,)
    print(f"dtype  : {emb.dtype}")        # expect float32
    print(f"L2 norm: {np.linalg.norm(emb):.2f}")
