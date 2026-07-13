# SimilarityX AI — Study Notes
## UI Iteration · Step 3 — Catalog Filters

> Revision note for college evaluation. Adds user control over the retrieval surface.

---

### What changed
The sidebar now supports filters for:
- category
- gender
- color
- same-category mode for catalog queries

Filters are applied after similarity scores are computed, while scanning the ranked list. This means the
model still ranks by visual similarity, but the final Top-K respects the user's catalog constraints.

### Why this improves the system
A real visual search tool should let users narrow the results. Someone searching for a shirt may not
want shoes, even if the visual embedding finds a close pose/background match. Filters make the
recommender more controllable and easier to evaluate.

### Eval line
*"I added catalog filters so the recommender is not only model-driven but also user-controllable, which
is closer to how ecommerce visual search works in practice."*

---

### Interview Q&A
- **Do filters replace similarity?** No. Similarity ranks products first; filters constrain which ranked
  products can be shown.
- **Why same-category mode?** It lets us inspect whether the model can find visually similar items within
  the query's known product category.
- **Why disable same-category for uploads?** Uploaded images do not have ground-truth catalog category,
  so the UI asks the user to choose a category filter instead.
