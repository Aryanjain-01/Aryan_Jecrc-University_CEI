"""
preprocessing.py — Turn a raw image into a tensor ResNet50 expects.

This is the single, reusable translation layer used by BOTH the embedding
generator (Step 5) and the Streamlit app (Step 7), so preprocessing is defined
in exactly one place. If it lived in two places and drifted, the query would be
preprocessed differently from the catalog and every similarity score would be
subtly wrong — a classic, hard-to-spot bug.

Pipeline (see docs/study-notes/Phase1_Step3):
    load -> convert to RGB -> resize to 224x224 -> float32
         -> resnet50.preprocess_input (RGB->BGR, subtract ImageNet mean)

We deliberately call Keras' own `preprocess_input` instead of hand-rolling
normalization, so our inputs EXACTLY match how ResNet50 was trained.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Union

import numpy as np
from PIL import Image
from tensorflow.keras.applications.resnet50 import preprocess_input

from src.config import IMAGE_SIZE

# A "source" may be a file path, or an already-open PIL image, or a file-like
# object (e.g. a Streamlit upload). load_image handles all three.
ImageSource = Union[str, Path, Image.Image]


def load_image(source: ImageSource) -> Image.Image:
    """Open `source` and guarantee a 3-channel RGB image.

    Why force RGB: some catalog images are grayscale (1 channel) or PNGs with an
    alpha channel (RGBA, 4 channels). ResNet50 demands (H, W, 3); `.convert("RGB")`
    normalizes every input to exactly 3 channels and prevents shape-mismatch crashes.
    """
    img = source if isinstance(source, Image.Image) else Image.open(source)
    return img.convert("RGB")


def preprocess_image(source: ImageSource, target_size=IMAGE_SIZE) -> np.ndarray:
    """Return a single preprocessed array of shape (224, 224, 3), float32.

    Steps:
        1. load + force RGB
        2. resize to the ResNet50/ImageNet input size (224x224)
        3. cast to float32 (preprocess_input needs floats, not uint8)
        4. resnet50.preprocess_input -> RGB->BGR + subtract ImageNet mean
    The result is NOT batched; callers add the batch dimension (or use
    preprocess_batch) so batching logic lives in one obvious place.
    """
    img = load_image(source).resize(target_size)          # (224, 224, 3), uint8
    arr = np.asarray(img, dtype=np.float32)               # -> float32
    arr = preprocess_input(arr)                           # ImageNet caffe-mode normalization
    return arr


def preprocess_batch(sources: Iterable[ImageSource], target_size=IMAGE_SIZE) -> np.ndarray:
    """Preprocess many images into one (N, 224, 224, 3) float32 tensor.

    Stacking into a single tensor is what lets the network score a whole batch
    in one forward pass (much faster than one-at-a-time).
    """
    batch = [preprocess_image(s, target_size) for s in sources]
    return np.stack(batch, axis=0)


if __name__ == "__main__":
    # Smoke test: preprocess the first subset image and report the tensor stats.
    import pandas as pd
    from src.config import SUBSET_CSV, PROJECT_ROOT

    df = pd.read_csv(SUBSET_CSV)
    sample_path = PROJECT_ROOT / df.iloc[0]["image_path"]
    x = preprocess_image(sample_path)
    print(f"sample: {sample_path.name}")
    print(f"shape : {x.shape}")          # expect (224, 224, 3)
    print(f"dtype : {x.dtype}")          # expect float32
    print(f"range : [{x.min():.1f}, {x.max():.1f}]  (mean-subtracted, so ~[-124, 151])")
