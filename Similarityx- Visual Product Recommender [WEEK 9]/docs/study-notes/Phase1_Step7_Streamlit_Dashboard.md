# SimilarityX AI — Study Notes
## Phase 1 · Step 7 — Streamlit Dashboard

> Revision note for college evaluation.

---

### What Streamlit is
Turns a plain Python script into an interactive web app — no HTML/CSS/JS. You write `st.title`,
`st.image`, `st.file_uploader` and it renders a browser UI. Exists so ML engineers can DEMO models
fast without web-dev. Standard for ML prototypes -> perfect for the eval demo.

### THE concept to explain: reruns + caching
Streamlit's execution model: **every user interaction re-runs the entire script top to bottom.**
Simple & predictable, but a trap — loading the Recommender means loading TensorFlow + ResNet50
weights + the 33 MB embeddings file. On every rerun that would cost seconds per click.

**Fix: `@st.cache_resource`** — "build this expensive object ONCE, reuse across all reruns." We wrap
the Recommender load in it -> model + index load a single time at startup; every query is instant.
This is THE Streamlit ML pattern.
- `@st.cache_resource` -> for unserializable/expensive objects (models, DB connections, our Recommender).
- `@st.cache_data` -> for data (DataFrames, arrays); returns a copy each call. We use it for the
  catalog CSV.

Likely eval Q: *"How do you avoid reloading the model on every interaction?"* -> `@st.cache_resource`.

### Features built (the brief)
- Upload image (jpg/jpeg/png) OR "Surprise me" random catalog item.
- Preview the query image.
- Top-K similar products in a grid (`st.columns`), each with:
  - similarity score
  - product name + category + metadata (gender, colour)
- Inference time (ms) shown as a caption.
- Sidebar K slider (3-10) + a "how it works" explainer.

### Run
```
cd ~/Documents/similarityx
streamlit run app/streamlit_app.py
```
Opens http://localhost:8501. First load builds the cached Recommender (a few seconds); after that
each query is instant.

### Flow recap (end-to-end, say this in the demo)
upload -> PIL RGB -> preprocess_image (224x224 + ImageNet norm) -> frozen ResNet50 -> 2048-d embedding
-> L2-normalize -> dot product vs 3,964 normalized catalog vectors (= cosine) -> argsort -> Top-K
-> render with scores + metadata + timing.

### Interview Q&A
- **Why Streamlit?** fast pure-Python ML UI; no web-dev; standard for prototypes/demos.
- **Execution model?** whole script re-runs on each interaction.
- **How avoid reloading the model?** `@st.cache_resource` (load once, reuse).
- **cache_resource vs cache_data?** resource = expensive objects/models; data = serializable data
  (returns a copy).

### Common mistakes
- Not caching the model -> app reloads TF on every click (slow).
- Loading embeddings inside the request path instead of once at startup.
- Passing raw bytes without `.convert("RGB")` (handled inside our preprocessing).
