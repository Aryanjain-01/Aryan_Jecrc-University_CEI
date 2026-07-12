"""
evaluate.py — Baseline vs Siamese retrieval comparison (Iteration 2, Steps 7 & 8).

Computes Precision@K and Recall@K for BOTH indexes under an identical protocol,
times a query on each, prints a comparison table, and saves a side-by-side visual
of one query's Top-5 under baseline vs improved embeddings.

Relevance rule: a retrieved item is "relevant" if it shares the query's category.
Protocol (identical for both models): every catalog item is a query; retrieve its
Top-K from the same index EXCLUDING itself; average P@K / R@K over all queries.

Run (after embeddings.py and improved_embeddings.py):
    python -m src.evaluate
"""

from __future__ import annotations

import pickle
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from src.config import EMBEDDINGS_DIR, FIGURES_DIR, PROJECT_ROOT

# Path constants defined here directly so this module needs NO TensorFlow.
BASELINE_PKL = EMBEDDINGS_DIR / "embeddings.pkl"
IMPROVED_PKL = EMBEDDINGS_DIR / "improved_embeddings.pkl"
KS = (1, 5, 10)


def _load(pkl):
    with open(pkl, "rb") as f:
        return pickle.load(f)


def _normalize(x):
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-8, None)


def _topk(Xn, i, k):
    """Top-k neighbor indices of row i (excluding itself), most similar first."""
    sims = Xn @ Xn[i]
    sims[i] = -1e9                                   # exclude self
    part = np.argpartition(-sims, k)[:k]
    return part[np.argsort(-sims[part])], sims


def evaluate_index(embeddings, categories, ks=KS, seed=42):
    """Return (precision dict, recall dict, avg_query_ms, dim)."""
    Xn = _normalize(embeddings.astype("float32"))
    n = len(Xn)
    maxk = max(ks)
    cat_total = {c: int(np.sum(categories == c)) for c in np.unique(categories)}
    P = {k: 0.0 for k in ks}
    R = {k: 0.0 for k in ks}

    t0 = time.time()
    for i in range(n):
        order, _ = _topk(Xn, i, maxk)
        rel = (categories[order] == categories[i]).astype(int)
        total_rel = max(cat_total[categories[i]] - 1, 1)   # exclude the query itself
        for k in ks:
            hits = int(rel[:k].sum())
            P[k] += hits / k
            R[k] += hits / total_rel
    elapsed = time.time() - t0

    P = {k: P[k] / n for k in ks}
    R = {k: R[k] / n for k in ks}
    return P, R, elapsed / n * 1000.0, Xn.shape[1]


def save_comparison_figure(base, imp, query_idx, k=5,
                           out=FIGURES_DIR / "baseline_vs_siamese.png"):
    """Side-by-side Top-k for one query: baseline row vs Siamese row."""
    paths = base["image_paths"]
    meta = base["metadata"]
    Xb = _normalize(base["embeddings"].astype("float32"))
    Xi = _normalize(imp["embeddings"].astype("float32"))
    b_order, _ = _topk(Xb, query_idx, k)
    i_order, _ = _topk(Xi, query_idx, k)

    def img(idx):
        return Image.open(PROJECT_ROOT / paths[idx]).convert("RGB")

    fig, axes = plt.subplots(2, k + 1, figsize=((k + 1) * 2, 4.6))
    q_cat = meta.iloc[query_idx]["category"]
    for r in range(2):
        axes[r, 0].imshow(img(query_idx)); axes[r, 0].axis("off")
    axes[0, 0].set_title(f"QUERY\n[{q_cat}]", fontsize=10)

    for j, idx in enumerate(b_order):
        axes[0, j + 1].imshow(img(idx)); axes[0, j + 1].axis("off")
        axes[0, j + 1].set_title(meta.iloc[idx]["category"], fontsize=9)
    for j, idx in enumerate(i_order):
        axes[1, j + 1].imshow(img(idx)); axes[1, j + 1].axis("off")
        axes[1, j + 1].set_title(meta.iloc[idx]["category"], fontsize=9)

    fig.text(0.005, 0.72, "BASELINE", rotation=90, va="center", fontsize=11, weight="bold")
    fig.text(0.005, 0.28, "SIAMESE", rotation=90, va="center", fontsize=11, weight="bold")
    fig.suptitle("Top-5 recommendations: Baseline (2048-d) vs Siamese (128-d)", fontsize=12)
    fig.tight_layout(rect=[0.02, 0, 1, 0.96])
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"[ok] comparison figure saved -> {out}")


def main():
    base = _load(BASELINE_PKL)
    imp = _load(IMPROVED_PKL)
    categories = base["metadata"]["category"].to_numpy()

    print("Evaluating BASELINE (2048-d) ...")
    Pb, Rb, ms_b, dim_b = evaluate_index(base["embeddings"], categories)
    print("Evaluating SIAMESE  (128-d) ...")
    Pi, Ri, ms_i, dim_i = evaluate_index(imp["embeddings"], categories)

    print("\n================ Baseline vs Siamese ================")
    print(f"{'metric':<14}{'baseline':>12}{'improved':>12}{'Δ':>10}")
    for k in KS:
        print(f"{'Precision@'+str(k):<14}{Pb[k]:>12.3f}{Pi[k]:>12.3f}{Pi[k]-Pb[k]:>+10.3f}")
    for k in KS:
        print(f"{'Recall@'+str(k):<14}{Rb[k]:>12.4f}{Ri[k]:>12.4f}{Ri[k]-Rb[k]:>+10.4f}")
    print(f"{'query ms':<14}{ms_b:>12.3f}{ms_i:>12.3f}")
    print(f"{'dim':<14}{dim_b:>12}{dim_i:>12}")
    print("=====================================================")

    # Visual for one representative query (fixed seed -> reproducible).
    rng = np.random.default_rng(7)
    save_comparison_figure(base, imp, int(rng.integers(len(categories))))


if __name__ == "__main__":
    main()
