"""
data_prep.py — Build a balanced, reproducible subset of the Fashion Product
Images (small) dataset for SimilarityX AI.

Pipeline (Phase 1, Step 2):
    1. Load styles.csv robustly (skip the malformed comma-in-title rows).
    2. Assign each product to one of 8 target categories (drop the rest).
    3. Keep only products whose image file actually exists on disk.
    4. Sample a fixed number per category (reproducible via RANDOM_STATE).
    5. Copy the selected images into data/subset/images/.
    6. Save subset.csv + diagnostic figures (class balance + a sample grid).

Run from the project root:
    python -m src.data_prep
"""

from __future__ import annotations

import shutil

import matplotlib
matplotlib.use("Agg")                 # non-interactive backend: save PNGs, no GUI
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

from src.config import (
    PROJECT_ROOT,
    RAW_STYLES_CSV,
    RAW_IMAGES_DIR,
    SUBSET_DIR,
    SUBSET_IMAGES_DIR,
    SUBSET_CSV,
    FIGURES_DIR,
    CATEGORY_MAP,
    IMAGES_PER_CATEGORY,
    RANDOM_STATE,
)

# Columns we actually need downstream. Everything else in styles.csv is either
# display metadata or unused, but we keep the useful display fields too.
ESSENTIAL_COLUMNS = ["id", "articleType", "productDisplayName"]
DISPLAY_COLUMNS = ["gender", "baseColour", "season", "year", "usage"]


def load_styles(csv_path=RAW_STYLES_CSV) -> pd.DataFrame:
    """Load styles.csv defensively.

    The classic bug in this dataset: `productDisplayName` sometimes contains
    commas, so those rows have more fields than headers and break a naive read.
    `on_bad_lines="skip"` drops that handful of rows (negligible loss). We then
    enforce essential columns, cast the id to int, and de-duplicate on id.
    """
    df = pd.read_csv(csv_path, on_bad_lines="skip")
    df = df.dropna(subset=ESSENTIAL_COLUMNS)
    df["id"] = df["id"].astype(int)
    df = df.drop_duplicates(subset="id").reset_index(drop=True)
    return df


def assign_category(article_type: str):
    """Map a fine-grained `articleType` to one of our 8 buckets, else None."""
    for bucket, article_types in CATEGORY_MAP.items():
        if article_type in article_types:
            return bucket
    return None


def image_path_for(product_id: int):
    """Absolute path to a raw image given its product id."""
    return RAW_IMAGES_DIR / f"{product_id}.jpg"


def build_subset() -> pd.DataFrame:
    """Create the balanced subset and persist it to disk. Returns the subset."""
    df = load_styles()

    # 2) Assign our category label; drop products outside the 8 buckets.
    df["category"] = df["articleType"].apply(assign_category)
    df = df[df["category"].notna()].copy()

    # 3) Keep only rows whose image file actually exists (id <-> image join).
    df = df[df["id"].apply(lambda i: image_path_for(i).is_file())].copy()

    # 4) Balanced, reproducible sampling per category.
    sampled_frames = []
    for bucket in CATEGORY_MAP:
        pool = df[df["category"] == bucket]
        n = min(IMAGES_PER_CATEGORY, len(pool))
        if n < IMAGES_PER_CATEGORY:
            print(f"[warn] '{bucket}': only {len(pool)} images available "
                  f"(< {IMAGES_PER_CATEGORY}); taking all of them.")
        sampled_frames.append(pool.sample(n=n, random_state=RANDOM_STATE))
    subset = pd.concat(sampled_frames).reset_index(drop=True)

    # 5) Copy chosen images into the subset folder; record a relative path.
    SUBSET_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    relative_paths = []
    for pid in subset["id"]:
        dst = SUBSET_IMAGES_DIR / f"{pid}.jpg"
        if not dst.exists():
            shutil.copy(image_path_for(pid), dst)
        relative_paths.append(str(dst.relative_to(PROJECT_ROOT)))
    subset["image_path"] = relative_paths

    # 6) Persist the subset table.
    keep = ESSENTIAL_COLUMNS + DISPLAY_COLUMNS + ["category", "image_path"]
    keep = [c for c in keep if c in subset.columns]
    SUBSET_DIR.mkdir(parents=True, exist_ok=True)
    subset[keep].to_csv(SUBSET_CSV, index=False)
    print(f"[ok] subset saved: {len(subset)} rows -> {SUBSET_CSV}")
    return subset


def plot_class_balance(subset: pd.DataFrame) -> None:
    """Bar chart proving the subset is balanced across the 8 categories."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    counts = subset["category"].value_counts().reindex(CATEGORY_MAP.keys())
    plt.figure(figsize=(9, 5))
    plt.bar(counts.index, counts.values, color="#4C7EF3")
    plt.title("Subset class balance (images per category)")
    plt.ylabel("count")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    out = FIGURES_DIR / "class_balance.png"
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"[ok] figure saved -> {out}")


def plot_sample_grid(subset: pd.DataFrame, per_category: int = 3) -> None:
    """Grid of a few real images per category — sanity-checks the label map."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    categories = list(CATEGORY_MAP.keys())
    rows, cols = len(categories), per_category
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.2))
    for r, bucket in enumerate(categories):
        picks = subset[subset["category"] == bucket].head(per_category)
        for c in range(cols):
            ax = axes[r, c]
            ax.axis("off")
            if c < len(picks):
                img_path = PROJECT_ROOT / picks.iloc[c]["image_path"]
                ax.imshow(Image.open(img_path).convert("RGB"))
                if c == 0:
                    ax.set_title(bucket, loc="left", fontsize=10)
    fig.suptitle("Sample images per category (label sanity check)")
    fig.tight_layout()
    out = FIGURES_DIR / "sample_grid.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"[ok] figure saved -> {out}")


def main() -> None:
    subset = build_subset()
    plot_class_balance(subset)
    plot_sample_grid(subset)


if __name__ == "__main__":
    main()
