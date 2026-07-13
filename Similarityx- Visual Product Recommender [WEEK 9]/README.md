---
title: SimilarityX AI
emoji: 🛍️
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.36.0
app_file: app/streamlit_app.py
pinned: false
---

# SimilarityX AI
**AI-Powered Visual Product Recommendation Engine using Deep Metric Learning**

Given a fashion product image, retrieve the most visually similar products in the catalog.
The project evolves in two iterations, mirroring how production ML systems mature:

- **Iteration 1 — Baseline:** pretrained ResNet50 feature extractor + cosine-similarity retrieval.
- **Iteration 2 — Advanced:** fine-tuned encoder + Siamese network trained with metric learning
  (contrastive / triplet loss) for tighter, cleaner embeddings.

## Project structure
```
similarityx/
├── data/
│   ├── raw/        # downloaded Kaggle dataset (styles.csv + images/) — READ-ONLY
│   └── subset/     # generated balanced ~4k subset (subset.csv + images/)
├── src/            # reusable library
│   ├── config.py           # single source of truth for paths & constants
│   ├── data_prep.py        # Step 2: build the balanced subset
│   ├── preprocessing.py    # Step 3
│   ├── feature_extractor.py# Step 4
│   ├── embeddings.py       # Step 5
│   └── recommender.py      # Step 6
├── artifacts/      # generated outputs (embeddings, figures)
├── app/            # Streamlit dashboard (Step 7)
├── docs/study-notes/  # deep-dive revision notes per step
└── requirements.txt
```

## Setup
```bash
pip install -r requirements.txt

# Kaggle dataset (small version):
#   place kaggle.json in ~/.kaggle/  (Kaggle -> Settings -> Create New Token)
kaggle datasets download -d paramaggarwal/fashion-product-images-small -p data/raw --unzip
# result: data/raw/styles.csv and data/raw/images/*.jpg
```

## Build the subset (Step 2)
```bash
python -m src.data_prep
```
Produces `data/subset/subset.csv`, copies the chosen images, and writes
`artifacts/figures/class_balance.png` and `sample_grid.png`.

## Tech stack
TensorFlow/Keras · OpenCV · Pillow · NumPy · Pandas · scikit-learn · Streamlit · Matplotlib
