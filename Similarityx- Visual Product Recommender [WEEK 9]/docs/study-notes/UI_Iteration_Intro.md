# SimilarityX AI — Study Notes
## UI Iteration · Intro — Turning the Model Demo into a Product Experience

> Revision note for college evaluation. The "why UI iteration exists" story.

---

### The limitation we're fixing
The model can retrieve similar products, but the first dashboard version presents results like a
technical demo: image, score, name, caption. That proves the pipeline works, but it does not yet feel
like a product recommendation engine a shopper or evaluator would trust.

The UI iteration improves the user-facing layer without changing the model. Same embeddings, same
retrieval logic, better communication.

### Why UI matters in an AI recommender
Recommendation quality is not only the algorithm. The user also needs to understand:
- what image was used as the query
- which products are ranked highest
- how confident the match is
- what metadata explains the result
- whether filters or model choices changed the output

A polished UI makes the AI system easier to inspect, compare, and defend during evaluation.

### The plan (Steps)
1. Upgrade result cards: rank, product image, score meter, and metadata tags.
2. Add query-vs-results comparison layout.
3. Add catalog filters: category, gender, color, and same-category mode.
4. Add baseline vs improved model toggle.
5. Add comparison/evaluation mode for baseline vs Siamese results.

Eval line: *"After building the retrieval model, I improved the UI as a separate iteration so the
system behaves like an inspectable recommendation product, not only a backend ML demo."*

---

### Interview Q&A
- **Why do a UI iteration after model work?** The model output must be understandable and testable by
  users; UI turns raw nearest-neighbor results into a usable recommender.
- **Did the UI change model accuracy?** No. It improves presentation and inspection; the retrieval
  logic stays the same unless a later step explicitly compares models.
- **Why show metadata tags?** Tags help users verify whether the visual match also agrees with
  category, gender, or color.
