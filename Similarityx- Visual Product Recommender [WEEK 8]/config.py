"""
config.py — Central configuration for SimilarityX AI.

Every path and tunable constant lives here so the rest of the codebase never
hard-codes a location. All paths are derived from the project root, which makes
the project portable: it works no matter what the absolute folder name is or
which machine it runs on.

Interview note: centralizing configuration is a standard software-engineering
practice (single source of truth). To relocate the project or change the subset
size, you edit ONE file — nothing else.
"""

from pathlib import Path

# --------------------------------------------------------------------------- #
# Project layout
# --------------------------------------------------------------------------- #
# __file__ is .../SimilarityX AI/src/config.py
# parents[1] climbs one level from /src up to the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# --- Raw data: the downloaded Kaggle "small" dataset. Treated as READ-ONLY. ---
RAW_DIR = PROJECT_ROOT / "data" / "raw"
RAW_STYLES_CSV = RAW_DIR / "styles.csv"
RAW_IMAGES_DIR = RAW_DIR / "images"

# --- Balanced subset we build in Step 2 (generated, safe to delete/rebuild). --
SUBSET_DIR = PROJECT_ROOT / "data" / "subset"
SUBSET_IMAGES_DIR = SUBSET_DIR / "images"
SUBSET_CSV = SUBSET_DIR / "subset.csv"

# --- Generated artifacts (embeddings, figures). Reproducible outputs. --------
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
EMBEDDINGS_DIR = ARTIFACTS_DIR / "embeddings"
FIGURES_DIR = ARTIFACTS_DIR / "figures"

# --------------------------------------------------------------------------- #
# Subset construction
# --------------------------------------------------------------------------- #
IMAGES_PER_CATEGORY = 500          # target count per category
RANDOM_STATE = 42                  # fixes the random sample -> reproducible

# 8 target categories -> the fine-grained `articleType` values that fill each
# bucket. Two buckets are intentionally NOT one-to-one with articleType:
#   * "Bags"  : we use "Handbags" for a visually coherent bucket.
#   * "Shoes" : NOT a single articleType — it is a super-category spanning
#               Casual/Sports/Formal Shoes. This is a deliberate design choice
#               we can defend in the eval (it mirrors how a store groups them).
CATEGORY_MAP = {
    "Shirts":  ["Shirts"],
    "Jeans":   ["Jeans"],
    "Kurtas":  ["Kurtas"],
    "Dresses": ["Dresses"],
    "Sandals": ["Sandals"],
    "Watches": ["Watches"],
    "Bags":    ["Handbags"],
    "Shoes":   ["Casual Shoes", "Sports Shoes", "Formal Shoes"],
}

# --------------------------------------------------------------------------- #
# Image preprocessing (used from Step 3 onward)
# --------------------------------------------------------------------------- #
IMAGE_SIZE = (224, 224)            # ResNet50 / ImageNet default input size
