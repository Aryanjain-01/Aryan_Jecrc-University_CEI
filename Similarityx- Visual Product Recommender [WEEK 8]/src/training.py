"""
training.py — Train the projection head with triplet loss (Iteration 2, Step 5).

Brings everything together:
    embeddings.pkl  -> TripletSampler  -> (A, P, N) triplets
    build_siamese_triplet_model (shared head)
    compile with triplet_loss + triplet_accuracy
    fit -> the head learns a fashion-tuned space
    save the trained head for Step 6.

Only the small head is trained (the ResNet50 backbone is frozen and never touched),
and it trains on the PRECOMPUTED 2048-d embeddings, so this runs in ~a minute on CPU.

Run:
    python -m src.training
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

from src.config import ARTIFACTS_DIR
from src.embeddings import load_embeddings
from src.triplets import TripletSampler
from src.siamese import build_projection_head, build_siamese_triplet_model
from src.losses import triplet_loss, triplet_accuracy

# --- Paths ---
MODELS_DIR = ARTIFACTS_DIR / "models"
HEAD_PATH = MODELS_DIR / "projection_head.keras"

# --- Hyperparameters (see docs/study-notes/Iteration2_Step5) ---
MARGIN = 0.2            # required gap between d(A,P) and d(A,N)
LEARNING_RATE = 1e-3    # normal LR: the head is trained from scratch (not fine-tuned)
EPOCHS = 30
BATCH_SIZE = 128
N_TRIPLETS = 20_000     # size of the sampled training set
VAL_SPLIT = 0.1
SEED = 42


def train():
    tf.random.set_seed(SEED)
    np.random.seed(SEED)

    # 1) Load precomputed catalog embeddings + their category labels.
    data = load_embeddings()
    embeddings = data["embeddings"]
    categories = data["metadata"]["category"].to_numpy()

    # 2) Sample a large, varied set of (anchor, positive, negative) triplets.
    sampler = TripletSampler(embeddings, categories, seed=SEED)
    (A, P, N), y = sampler.sample_batch(N_TRIPLETS)
    print(f"Sampled {N_TRIPLETS} triplets. A={A.shape}")

    # 3) Build the shared-weight Siamese model and compile it.
    head = build_projection_head()
    siamese, _ = build_siamese_triplet_model(head)
    siamese.compile(
        optimizer=Adam(LEARNING_RATE),
        loss=triplet_loss(MARGIN),
        metrics=[triplet_accuracy],
    )

    # 4) Train. Watch: loss -> 0, triplet_accuracy -> ~1.0, val tracking train.
    print(f"Training: {EPOCHS} epochs, batch {BATCH_SIZE}, margin {MARGIN}, lr {LEARNING_RATE}\n")
    siamese.fit(
        [A, P, N], y,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_split=VAL_SPLIT,
        verbose=2,
    )

    # 5) Save ONLY the trained head (Step 6 composes ResNet50 -> head).
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    head.save(HEAD_PATH)
    print(f"\n[ok] trained projection head saved -> {HEAD_PATH}")


if __name__ == "__main__":
    train()
