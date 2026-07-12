# SimilarityX AI — Study Notes
## UI Iteration · Step 4 — Baseline vs Siamese Model Toggle

> Revision note for college evaluation. Connects the dashboard to Iteration 2.

---

### What changed
The dashboard can now run recommendations using either:
- **Baseline ResNet50 (2048-d)** embeddings
- **Siamese improved (128-d)** embeddings

The same `Recommender` class now accepts a model name and loads the correct embedding index. For the
improved model, query images are passed through the trained projection head so query vectors and catalog
vectors live in the same 128-d metric-learning space.

### Why this improves the system
Iteration 2 is only convincing if users can see its effect. The toggle makes the advanced model visible
inside the product demo, not only in offline metrics or saved figures.

### Eval line
*"I connected the UI to both embedding spaces, so the evaluator can switch between the ImageNet baseline
and the Siamese metric-learning model directly in the dashboard."*

---

### Interview Q&A
- **Why can't improved catalog embeddings be compared to baseline query embeddings?** They live in
  different vector spaces; the query must also pass through the projection head.
- **What changed in query time?** Baseline uses ResNet50 output directly; improved uses ResNet50 output
  followed by the trained projection head.
- **Why mention dimensions?** It shows an engineering benefit: 128-d vectors are much smaller than
  2048-d vectors.
