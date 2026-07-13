# SimilarityX AI — Study Notes
## UI Iteration · Step 1 — Product Cards for Ranked Recommendations

> Revision note for college evaluation. First visible UI upgrade.

---

### What changed
The old result view used plain Streamlit columns: image, similarity score, product name, and a caption.
Step 1 replaces that with ecommerce-style product cards:
- rank badge (`#1`, `#2`, ...)
- square product image area
- product name
- similarity score
- visual progress meter
- metadata tags for category, gender, and color

### Why this improves the system
Nearest-neighbor retrieval is a ranked task, so the UI should make ranking obvious. A rank badge tells
the user which item is the strongest match. A score meter makes the numeric cosine similarity easier to
read at a glance. Tags expose the catalog metadata so the evaluator can quickly check whether results
are visually and semantically reasonable.

### Implementation note
The cards are rendered as HTML/CSS inside Streamlit. Catalog images are encoded as small data URIs so
each card can contain the product image, rank, score, and tags in one consistent block.

### Eval line
*"I upgraded the recommendation output from plain columns to ranked product cards, which makes the
retrieval result easier to scan and closer to a real ecommerce visual-search interface."*

---

### Interview Q&A
- **Why show rank and score?** Rank communicates retrieval order; score communicates similarity
  strength.
- **Why tags?** Tags make it easier to validate whether the visual result also matches catalog
  metadata like category or color.
- **Did this affect model output?** No. It only changes presentation; the same Top-K results are shown.

### Common mistakes
- Showing images without rank, making retrieval order unclear.
- Showing only raw scores with no visual cue.
- Hiding metadata, which makes recommendation quality harder to inspect.
