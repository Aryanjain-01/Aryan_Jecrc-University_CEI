# SimilarityX AI — Study Notes
## Iteration 2 · Steps 7 & 8 — Retrieval + Baseline vs Siamese Comparison

> Revision note for college evaluation. Where we PROVE the improvement with numbers.

---

### Ground truth
A retrieved item is "relevant" if it shares the query's CATEGORY (same proxy as training). Coarse but
CONSISTENT — applied identically to both models, which is what a fair comparison needs.

### Precision@K
**P@K = (relevant items in top-K) / K.** "Of the K things I showed, what fraction were good?"
Example: query Watch, top-5 = [Watch,Watch,Bag,Watch,Watch] -> 4/5 = **P@5 = 0.80**.
Headline metric for a recommender (measures cleanliness of the results row the user sees).

### Recall@K
**R@K = (relevant in top-K) / (total relevant in catalog).** "Of all I could have found, what fraction
did I surface?"
Caveat: ~500 items/category -> perfect top-5 gives R@5 ~ 5/499 ~ 0.01 (tiny BY CONSTRUCTION). So
**Precision@K is the headline; Recall@K informative at larger K (e.g. R@100).** Report both, lean on P@K.

### Evaluation protocol (identical for both models)
1. Every catalog item (or a large sample) = a query.
2. Retrieve top-K from the same index, EXCLUDING itself.
3. Count same-category hits -> P@K, R@K per query.
4. Average over all queries.
5. Run identically on baseline (2048-d) and improved (128-d); compare.
"Identical" = same queries, K, relevance rule; only the space differs -> valid claim, not confound.

### Other comparisons
- **Inference time / dim:** improved 128-d vs baseline 2048-d = 16x smaller -> faster search, less
  memory. Time a query on each.
- **Visual:** top-5 baseline vs top-5 improved side by side for a sample query (see fewer background/
  pose false matches).

### What success looks like
Improved P@K noticeably higher + smaller/faster vectors + cleaner visual row. If only marginally
higher, still a legitimate HONEST result (baseline already strong on clean studio photos; labels coarse)
— the point is you MEASURED it properly.

---

### Interview Q&A
- **Define P@K and R@K.** P@K = relevant in top-K / K; R@K = relevant in top-K / total relevant.
- **Why P@K headline not R@K here?** categories ~500 items -> R@small-K tiny by construction; precision
  measures user-facing quality.
- **How is the comparison fair?** identical queries, K, relevance rule; only the embedding space changes.
- **Besides quality, what improved?** 128-d vs 2048-d -> faster search, less memory.

### Common mistakes
- Not excluding the query itself -> inflates precision (self is always #1).
- Different query sets / K between models -> invalid comparison.
- Reporting R@5 as bad news without noting it's small by construction for large categories.
