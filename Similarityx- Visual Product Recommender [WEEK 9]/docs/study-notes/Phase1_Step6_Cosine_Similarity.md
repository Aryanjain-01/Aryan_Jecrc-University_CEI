# SimilarityX AI — Study Notes
## Phase 1 · Step 6 — Cosine Similarity & Top-5 Retrieval

> Revision note for college evaluation.

---

### Formula
```
cosine_similarity(A,B) = (A . B) / (||A|| * ||B||)
                       = sum_i(Ai*Bi) / (sqrt(sum Ai^2) * sqrt(sum Bi^2))
```
Dot product divided by the product of magnitudes. The division cancels magnitude -> pure direction.

### Geometric intuition (say this out loud)
Cosine similarity = **cosine of the angle between two vectors**. Picture arrows from the origin:
- same direction -> 0deg -> cos = **1** (max similar)
- perpendicular -> 90deg -> cos = **0** (unrelated)
- opposite -> 180deg -> cos = **-1** (max dissimilar)
"Do these two images point the same way in feature space?" Sandal~Sandal = small angle (~1);
Sandal~Watch = large angle (~0).
Note: ResNet50 features are post-ReLU (non-negative) -> all vectors in the positive orthant ->
our cosines land in ~[0,1], not the negative range.

### Mathematical intuition
Dot product sum_i(Ai*Bi) is large when both vectors are high in the SAME dimensions (same features
present). But raw dot product is inflated by magnitude ("loud" images). Dividing by magnitudes asks
"what FRACTION of feature-energy points the same way?" = fair, scale-free.
Example: A=[1,1,0], B=[2,2,0] -> Euclidean says far (sqrt2), cosine says **1.0** (same direction).
That gap is why we prefer cosine for embeddings.

### Advantages
- **Scale-invariant** — ignores magnitude, focuses on feature pattern (robust to brightness/loudness).
- **Bounded & interpretable** — in [-1,1] (here ~[0,1]); easy to show as a "similarity score."
- **Cheap & vectorizable** — pre-normalize catalog -> similarity = one matrix-vector multiply (ms).
- **Works in high-dim** — angle stays meaningful in 2048-d where raw Euclidean gets less discriminative.

### Limitations (BRIDGE TO ITERATION 2 — memorize)
- **Ignores magnitude entirely** — sometimes magnitude carries signal.
- **All 2048 dims weighted equally** — cosine doesn't know which features matter for fashion.
- **Deepest: the space isn't optimized for our task.** Vectors come from an ImageNet CLASSIFIER, not a
  fashion-similarity model. Cosine faithfully measures distance in a space organized for the WRONG
  objective -> sometimes returns ImageNet-texture matches, not good product matches.
  **No metric can fix a poorly-organized space — you must re-organize the space itself** = Deep Metric
  Learning / Siamese network (Iteration 2): train embeddings so similar products pull close, dissimilar
  push apart. Cosine on *pretrained* = strong baseline; cosine on *metric-learned* = the upgrade.

### Retrieval procedure (the code)
1. Load pickled catalog embeddings once.
2. Embed the query -> 2048 vector.
3. Cosine similarity vs all 3,964 catalog vectors.
4. Sort descending, take Top-K (K=5). If query IS a catalog item, its top hit is itself at 1.0 —
   detect & drop it so you see genuine neighbors.
5. Return each match's score + metadata + image path; time the query for the UI.

### cosine vs Euclidean (quick)
Euclidean sensitive to magnitude; cosine to direction. For L2-normalized vectors the two are
monotonically related (rank the same). We use cosine because embedding identity lives in direction.

---

### Interview Q&A
- **Define cosine similarity.** cos of angle between vectors = dot product / (product of magnitudes).
- **Geometric meaning?** 1=same direction, 0=perpendicular, -1=opposite.
- **Why cosine not Euclidean for embeddings?** scale-invariant; identity is in direction not magnitude.
- **Range here and why?** ~[0,1] because post-ReLU features are non-negative.
- **Biggest limitation?** measures a space organized for ImageNet, not fashion -> motivates metric
  learning (Iteration 2).

### Common mistakes
- Forgetting to exclude the query itself (self-match at 1.0) when the query is a catalog image.
- Claiming cosine "learns" similarity — it's a fixed metric; the *embeddings* must be learned to improve.
- Confusing scale-invariance (a feature) with ignoring important magnitude info (a limitation).
