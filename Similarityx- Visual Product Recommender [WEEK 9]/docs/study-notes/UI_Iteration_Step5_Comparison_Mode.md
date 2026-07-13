# SimilarityX AI — Study Notes
## UI Iteration · Step 5 — Side-by-Side Model Comparison

> Revision note for college evaluation. Makes the baseline-vs-improved story visible.

---

### What changed
The dashboard now has a comparison mode that runs both models for the same query, same K, and same
filters:
- left column: baseline ResNet50 results
- right column: Siamese improved results

This mirrors the offline evaluation protocol: keep query and settings identical, change only the
embedding space.

### Why this improves the system
Side-by-side results make the Iteration 2 claim inspectable. If the Siamese model is cleaner, the
evaluator can see fewer category/style mismatches. If the difference is small, the comparison is still
honest because both models were tested under identical conditions.

### Eval line
*"The comparison mode keeps query, K, and filters fixed while changing only the embedding model, so the
visual comparison is fair."*

---

### Interview Q&A
- **How is the comparison fair?** Same query, same K, same filters; only the embedding space changes.
- **Why show both in the UI if metrics already exist?** Metrics prove average behavior; side-by-side UI
  helps humans inspect concrete examples.
- **What should improve?** The Siamese model should return visually and categorically cleaner neighbors,
  while using smaller 128-d vectors.
