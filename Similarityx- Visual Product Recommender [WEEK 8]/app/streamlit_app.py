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

import os
import sys

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


@st.cache_resource(show_spinner="Loading model and catalog index (one time)…")
def load_recommender() -> Recommender:
    """Build the Recommender once and reuse across all reruns."""
    return Recommender()


@st.cache_data
def load_catalog() -> pd.DataFrame:
    """Subset table, cached — used for the 'surprise me' random example."""
    return pd.read_csv(SUBSET_CSV)


def show_results(results, elapsed_ms):
    """Render the Top-5 as a row of cards with score + metadata."""
    st.subheader(f"Top {len(results)} similar products")
    st.caption(f"Retrieved in {elapsed_ms:.1f} ms")
    cols = st.columns(len(results))
    for col, r in zip(cols, results):
        with col:
            st.image(str(PROJECT_ROOT / r["image_path"]), use_container_width=True)
            st.markdown(f"**{r['score']:.3f}** similarity")
            st.markdown(f"{r['productDisplayName']}")
            st.caption(
                f"{r['category']} · {r.get('gender','')} · {r.get('baseColour','')}"
            )


# --------------------------------------------------------------------------- #
# Layout
# --------------------------------------------------------------------------- #
st.title("🛍️ SimilarityX AI")
st.markdown(
    "**AI-powered visual product recommendation** — upload a fashion image and "
    "retrieve the most visually similar products using deep ResNet50 embeddings "
    "and cosine similarity."
)

recommender = load_recommender()
catalog = load_catalog()

k = st.sidebar.slider("Number of recommendations (K)", 3, 10, 5)
st.sidebar.markdown("---")
st.sidebar.markdown(
    "**How it works**\n\n"
    "1. Image → ResNet50 → 2048-d embedding\n"
    "2. Cosine similarity vs 3,964 catalog embeddings\n"
    "3. Return the Top-K nearest"
)

left, right = st.columns([1, 2])

with left:
    st.subheader("Query")
    uploaded = st.file_uploader("Upload a product image", type=["jpg", "jpeg", "png"])
    surprise = st.button("🎲 Surprise me (random catalog item)")

# Decide the query source: an upload, or a random catalog image.
query_image = None
exclude_id = None

if uploaded is not None:
    query_image = Image.open(uploaded).convert("RGB")
elif surprise:
    row = catalog.iloc[random.randint(0, len(catalog) - 1)]
    query_image = Image.open(PROJECT_ROOT / row["image_path"]).convert("RGB")
    exclude_id = int(row["id"])  # drop the query itself from results

with left:
    if query_image is not None:
        st.image(query_image, caption="Query image", use_container_width=True)

with right:
    if query_image is not None:
        with st.spinner("Finding similar products…"):
            results, elapsed_ms = recommender.recommend(
                query_image, k=k, exclude_id=exclude_id
            )
        show_results(results, elapsed_ms)
    else:
        st.info("⬅️ Upload an image or click **Surprise me** to see recommendations.")
