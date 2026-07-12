# SimilarityX AI — Study Notes
## UI Iteration · Step 2 — Query vs Results Comparison Layout

> Revision note for college evaluation. Makes the retrieval relationship visible.

---

### What changed
The dashboard now separates the interaction into two clear areas:
- **left:** the query product panel
- **right:** the ranked recommendation results

The query panel shows the input image like a product card, with source and metadata. For uploaded
images, it shows file details because the image is not part of the catalog. For random catalog samples,
it shows the known product name, category, gender, and color.

### Why this improves the system
Visual search is a comparison task. The evaluator should be able to look at the query and the returned
products at the same time. This layout makes it easier to judge whether the model is matching product
type, color, and style instead of accidentally matching pose or background.

The small flow row also explains the retrieval path:
**query image -> ResNet50 embedding -> Top-K matches.**

### Implementation note
The query image is converted to a data URI and rendered in a dedicated HTML/CSS panel. The result cards
from Step 1 stay unchanged, so Step 2 improves layout and interpretability without changing the
retrieval algorithm.

### Eval line
*"I redesigned the dashboard around a query-vs-results layout, because visual recommendation quality is
best evaluated by comparing the input image directly against the ranked outputs."*

---

### Interview Q&A
- **Why show the query as a product panel?** It makes the input and outputs visually comparable.
- **Why show uploaded image details instead of category?** Uploaded files are outside the catalog, so
  true metadata is unknown; showing filename, dimensions, and size is honest.
- **Did this change retrieval behavior?** No. The same recommender still returns Top-K matches; only the
  comparison layout changed.

### Common mistakes
- Showing results far away from the query image.
- Pretending uploaded images have catalog metadata.
- Explaining the model only in text instead of making the comparison visually obvious.
