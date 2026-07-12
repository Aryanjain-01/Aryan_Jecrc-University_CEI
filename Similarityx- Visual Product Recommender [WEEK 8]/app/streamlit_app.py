"""
streamlit_app.py — SimilarityX AI dashboard (Iteration 1 baseline).

Upload a fashion image -> see the Top-5 visually similar catalog products, each
with a similarity score and metadata, plus the query inference time.

Run from the project root:
    streamlit run app/streamlit_app.py

Key pattern: the Recommender (TensorFlow model + ResNet50 weights + 33 MB index)
is loaded ONCE via @st.cache_resource. Streamlit re-runs this whole script on
every interaction, so without caching each click would reload everything and
take seconds. With caching, load happens once and every query is instant.
"""

from __future__ import annotations

import base64
import html
import os
import sys
from io import BytesIO

# Streamlit runs this file directly, which puts app/ (not the project root) on
# sys.path — so `import src...` fails. Add the project root explicitly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random

import pandas as pd
import streamlit as st
from PIL import Image

from src.config import SUBSET_CSV, PROJECT_ROOT
from src.recommender import Recommender

st.set_page_config(page_title="SimilarityX AI", page_icon="🛍️", layout="wide")

st.markdown(
    """
    <style>
        .sx-results-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 1rem;
            margin-top: 0.75rem;
        }
        .sx-card {
            border: 1px solid rgba(49, 51, 63, 0.14);
            border-radius: 8px;
            overflow: hidden;
            background: #ffffff;
            box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04);
        }
        .sx-image-wrap {
            position: relative;
            aspect-ratio: 1 / 1;
            background: #f6f7f9;
            overflow: hidden;
        }
        .sx-image-wrap img {
            width: 100%;
            height: 100%;
            object-fit: contain;
            display: block;
        }
        .sx-rank {
            position: absolute;
            top: 0.5rem;
            left: 0.5rem;
            border-radius: 999px;
            background: rgba(17, 24, 39, 0.88);
            color: #ffffff;
            font-size: 0.78rem;
            font-weight: 700;
            line-height: 1;
            padding: 0.38rem 0.5rem;
        }
        .sx-body {
            padding: 0.85rem;
        }
        .sx-name {
            min-height: 2.6rem;
            color: #111827;
            font-size: 0.95rem;
            font-weight: 650;
            line-height: 1.3;
            margin-bottom: 0.65rem;
        }
        .sx-score-row {
            display: flex;
            justify-content: space-between;
            gap: 0.75rem;
            align-items: center;
            color: #374151;
            font-size: 0.82rem;
            font-weight: 600;
            margin-bottom: 0.35rem;
        }
        .sx-meter {
            height: 0.45rem;
            border-radius: 999px;
            background: #e5e7eb;
            overflow: hidden;
            margin-bottom: 0.7rem;
        }
        .sx-meter-fill {
            height: 100%;
            border-radius: inherit;
            background: linear-gradient(90deg, #0ea5e9, #22c55e);
        }
        .sx-tags {
            display: flex;
            flex-wrap: wrap;
            gap: 0.35rem;
        }
        .sx-tag {
            border: 1px solid #d1d5db;
            border-radius: 999px;
            color: #374151;
            background: #f9fafb;
            font-size: 0.72rem;
            line-height: 1;
            padding: 0.33rem 0.45rem;
        }
        .sx-query-card {
            border: 1px solid rgba(49, 51, 63, 0.14);
            border-radius: 8px;
            overflow: hidden;
            background: #ffffff;
            box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04);
            margin-top: 0.75rem;
        }
        .sx-query-image {
            aspect-ratio: 4 / 5;
            background: #f6f7f9;
            overflow: hidden;
        }
        .sx-query-image img {
            width: 100%;
            height: 100%;
            object-fit: contain;
            display: block;
        }
        .sx-query-body {
            padding: 0.95rem;
        }
        .sx-kicker {
            color: #0f766e;
            font-size: 0.76rem;
            font-weight: 750;
            text-transform: uppercase;
            margin-bottom: 0.35rem;
        }
        .sx-query-title {
            color: #111827;
            font-size: 1rem;
            font-weight: 700;
            line-height: 1.3;
            margin-bottom: 0.8rem;
        }
        .sx-detail-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.55rem;
        }
        .sx-detail {
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            background: #f9fafb;
            padding: 0.55rem;
        }
        .sx-detail-label {
            color: #6b7280;
            font-size: 0.68rem;
            font-weight: 700;
            text-transform: uppercase;
            margin-bottom: 0.18rem;
        }
        .sx-detail-value {
            color: #111827;
            font-size: 0.82rem;
            font-weight: 650;
            line-height: 1.25;
            overflow-wrap: anywhere;
        }
        .sx-flow {
            display: flex;
            align-items: center;
            gap: 0.65rem;
            color: #4b5563;
            font-size: 0.9rem;
            font-weight: 650;
            margin-top: 0.8rem;
            margin-bottom: 0.25rem;
        }
        .sx-flow-step {
            border: 1px solid #d1d5db;
            border-radius: 999px;
            background: #f9fafb;
            padding: 0.35rem 0.55rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading model and catalog index (one time)…")
def load_recommender(model_name: str) -> Recommender:
    """Build the Recommender once and reuse across all reruns."""
    return Recommender(model_name=model_name)


@st.cache_data
def load_catalog() -> pd.DataFrame:
    """Subset table, cached — used for the 'surprise me' random example."""
    return pd.read_csv(SUBSET_CSV)


def image_data_uri(image_path) -> str:
    """Encode a local catalog image so a recommendation card can render as HTML."""
    with open(PROJECT_ROOT / image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def pil_image_data_uri(image: Image.Image) -> str:
    """Encode an in-memory query image for the comparison panel."""
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def clean_value(value, fallback: str = "Unknown") -> str:
    """Return display-safe metadata while hiding empty/nan values."""
    if value is None:
        return fallback
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return fallback
    return text


def query_card_html(image: Image.Image, details: dict) -> str:
    """Render the query image and metadata as the left side of the comparison."""
    title = html.escape(clean_value(details.get("title"), "Uploaded product image"))
    source = html.escape(clean_value(details.get("source"), "Upload"))
    image_src = pil_image_data_uri(image)
    detail_items = [
        ("Source", source),
        ("Category", clean_value(details.get("category"))),
        ("Gender", clean_value(details.get("gender"))),
        ("Color", clean_value(details.get("colour"))),
    ]
    detail_html = "".join(
        '<div class="sx-detail">'
        f'<div class="sx-detail-label">{html.escape(label)}</div>'
        f'<div class="sx-detail-value">{html.escape(value)}</div>'
        '</div>'
        for label, value in detail_items
    )

    return (
        '<article class="sx-query-card">'
        f'<div class="sx-query-image"><img src="{image_src}" alt="{title}"></div>'
        '<div class="sx-query-body">'
        '<div class="sx-kicker">Visual query</div>'
        f'<div class="sx-query-title">{title}</div>'
        f'<div class="sx-detail-grid">{detail_html}</div>'
        '</div>'
        '</article>'
    )


def product_card_html(result: dict, rank: int) -> str:
    """Build one polished product card with rank, score meter, and metadata tags."""
    score = float(result["score"])
    score_pct = max(0, min(100, round(score * 100)))
    name = html.escape(str(result.get("productDisplayName", "Unknown product")))
    category = html.escape(str(result.get("category", "Unknown")))
    gender = html.escape(str(result.get("gender", "")))
    colour = html.escape(str(result.get("baseColour", "")))
    image_src = image_data_uri(result["image_path"])

    tags = [category]
    tags.extend(tag for tag in [gender, colour] if tag and tag.lower() != "nan")
    tags_html = "".join(f'<span class="sx-tag">{tag}</span>' for tag in tags)

    return (
        '<article class="sx-card">'
        '<div class="sx-image-wrap">'
        f'<img src="{image_src}" alt="{name}">'
        f'<span class="sx-rank">#{rank}</span>'
        '</div>'
        '<div class="sx-body">'
        f'<div class="sx-name">{name}</div>'
        '<div class="sx-score-row">'
        '<span>Similarity</span>'
        f'<span>{score:.3f}</span>'
        '</div>'
        f'<div class="sx-meter" aria-label="Similarity score {score:.3f}">'
        f'<div class="sx-meter-fill" style="width: {score_pct}%"></div>'
        '</div>'
        f'<div class="sx-tags">{tags_html}</div>'
        '</div>'
        '</article>'
    )


def show_results(results, elapsed_ms):
    """Render ranked results as polished ecommerce-style cards."""
    st.subheader(f"Top {len(results)} similar products")
    st.caption(f"Retrieved in {elapsed_ms:.1f} ms")
    if not results:
        st.warning("No products matched the current filters. Loosen a filter and try again.")
        return
    cards = "".join(product_card_html(r, i) for i, r in enumerate(results, 1))
    st.markdown(f'<section class="sx-results-grid">{cards}</section>', unsafe_allow_html=True)


def show_query_panel(image: Image.Image, details: dict):
    """Show the query as a comparable product panel."""
    st.subheader("Query product")
    st.markdown(query_card_html(image, details), unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Layout
# --------------------------------------------------------------------------- #
st.title("🛍️ SimilarityX AI")
st.markdown(
    "**AI-powered visual product recommendation** — upload a fashion image and "
    "retrieve the most visually similar products using deep ResNet50 embeddings "
    "and cosine similarity."
)

catalog = load_catalog()

MODEL_OPTIONS = {
    "Baseline ResNet50 (2048-d)": "baseline",
    "Siamese improved (128-d)": "improved",
}

k = st.sidebar.slider("Number of recommendations (K)", 3, 10, 5)
mode = st.sidebar.radio(
    "Recommendation mode",
    ["Single model", "Compare baseline vs improved"],
)
selected_model_label = st.sidebar.selectbox(
    "Model",
    list(MODEL_OPTIONS.keys()),
    disabled=mode == "Compare baseline vs improved",
)
st.sidebar.markdown("---")
st.sidebar.subheader("Filters")
category_filter = st.sidebar.selectbox("Category", ["Any"] + sorted(catalog["category"].dropna().unique()))
gender_filter = st.sidebar.selectbox("Gender", ["Any"] + sorted(catalog["gender"].dropna().unique()))
colour_filter = st.sidebar.selectbox("Color", ["Any"] + sorted(catalog["baseColour"].dropna().unique()))
same_category = st.sidebar.checkbox("Only same category as query")
st.sidebar.markdown("---")
st.sidebar.markdown(
    "**How it works**\n\n"
    "1. Image → embedding model\n"
    "2. Apply optional catalog filters\n"
    "3. Return the Top-K nearest products"
)

left, right = st.columns([1, 2])

with left:
    st.subheader("Start a search")
    uploaded = st.file_uploader("Upload a product image", type=["jpg", "jpeg", "png"])
    surprise = st.button("🎲 Surprise me (random catalog item)")

# Decide the query source: an upload, or a random catalog image.
query_image = None
exclude_id = None
query_details = {}
query_category = None

if uploaded is not None:
    query_image = Image.open(uploaded).convert("RGB")
    width, height = query_image.size
    query_details = {
        "source": "Uploaded image",
        "title": uploaded.name,
        "category": "Not in catalog",
        "gender": f"{width} x {height}px",
        "colour": f"{uploaded.size / 1024:.1f} KB",
    }
elif surprise:
    row = catalog.iloc[random.randint(0, len(catalog) - 1)]
    query_image = Image.open(PROJECT_ROOT / row["image_path"]).convert("RGB")
    exclude_id = int(row["id"])  # drop the query itself from results
    query_category = clean_value(row.get("category"), "")
    query_details = {
        "source": "Catalog sample",
        "title": row.get("productDisplayName"),
        "category": row.get("category"),
        "gender": row.get("gender"),
        "colour": row.get("baseColour"),
    }

with left:
    if query_image is not None:
        show_query_panel(query_image, query_details)

with right:
    if query_image is not None:
        filters = {}
        if category_filter != "Any":
            filters["category"] = category_filter
        if gender_filter != "Any":
            filters["gender"] = gender_filter
        if colour_filter != "Any":
            filters["baseColour"] = colour_filter
        if same_category and query_category:
            filters["category"] = query_category

        st.markdown(
            '<div class="sx-flow">'
            '<span class="sx-flow-step">Query image</span>'
            '<span>→</span>'
            '<span class="sx-flow-step">Embedding space</span>'
            '<span>→</span>'
            f'<span class="sx-flow-step">Top-{k} matches</span>'
            '</div>',
            unsafe_allow_html=True,
        )
        if same_category and not query_category:
            st.info("Same-category mode needs a catalog query. Use Surprise me or choose a category filter.")

        if mode == "Single model":
            model_name = MODEL_OPTIONS[selected_model_label]
            recommender = load_recommender(model_name)
            st.caption(f"Model: {recommender.label} · {recommender.embedding_dim}-d vectors")
            with st.spinner("Finding similar products…"):
                results, elapsed_ms = recommender.recommend(
                    query_image, k=k, exclude_id=exclude_id, filters=filters
                )
            show_results(results, elapsed_ms)
        else:
            base_col, improved_col = st.columns(2)
            with base_col:
                recommender = load_recommender("baseline")
                st.subheader("Baseline ResNet50")
                st.caption("2048-d ImageNet embedding space")
                with st.spinner("Running baseline…"):
                    results, elapsed_ms = recommender.recommend(
                        query_image, k=k, exclude_id=exclude_id, filters=filters
                    )
                show_results(results, elapsed_ms)
            with improved_col:
                recommender = load_recommender("improved")
                st.subheader("Siamese improved")
                st.caption("128-d metric-learning embedding space")
                with st.spinner("Running improved model…"):
                    results, elapsed_ms = recommender.recommend(
                        query_image, k=k, exclude_id=exclude_id, filters=filters
                    )
                show_results(results, elapsed_ms)
    else:
        st.info("⬅️ Upload an image or click **Surprise me** to see recommendations.")
