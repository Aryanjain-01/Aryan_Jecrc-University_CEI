# SimilarityX AI — Study Notes
## Iteration 2 · Step 3 — Building Pairs & Triplets

> Revision note for college evaluation.

---

### The key trick: labels -> similarity supervision
Siamese nets learn from COMPARISONS, not single labeled images. We manufacture examples from the
**category** column: **same category = "similar", different category = "dissimilar".** That one
assumption turns the labeled catalog into unlimited comparison examples. Standard when you only have
class labels.

**Limitation (name it):** category is a COARSE proxy for style. Two very different shirts get labeled
"similar" just because both are Shirts. Ideal = finer "same-style" labels. With only category labels
the model learns "same category = close" — big improvement over ImageNet space, but it's the ceiling.

### Positive pairs / negative pairs / triplets
- **Positive pair (anchor, positive):** should be CLOSE. Both from SAME category.
- **Negative pair (anchor, negative):** should be FAR. Different categories.
- **Triplet (anchor, positive, negative):** one anchor + same-cat positive + diff-cat negative =
  one training example for triplet loss.
Training on precomputed 2048-d embeddings -> a triplet is 3 saved vectors (emb[a], emb[p], emb[n])
pulled by row index from embeddings.pkl. No images touched during training.

### Sampling strategy (why sample, not enumerate)
#triplets ~ 4000 anchors x ~500 positives x ~3400 negatives ~ BILLIONS. Can't enumerate -> sample:
- per anchor: random positive (same cat) + random negative (diff cat).
- generate a few thousand FRESH triplets per epoch (resample each epoch = variety = regularizer;
  can't memorize a fixed list).

**Random vs hard (Step 4 preview):** random triplets are often TOO EASY (random shirt-vs-watch already
far -> loss ~0 -> no gradient). **Hard/semi-hard negative mining** picks negatives currently CLOSE to
the anchor (the confusing ones) -> that's where learning happens. Build random sampler now; mining
plugs on top in Step 4.

### Offline vs online
- **Offline:** pre-generate one fixed triplet list, reuse. Simple, low variety.
- **Online:** sample on the fly during training. More variety; required for hard mining (depends on
  current embeddings). We build an online-style generator.

### Keep evaluation fair
For Step 8, measure retrieval the SAME way for baseline and Siamese (category = ground-truth relevance,
ideally a held-out query set), else "it improved" is meaningless.

---

### Interview Q&A
- **Where does the similar/dissimilar label come from?** category column (same=positive, diff=negative);
  a proxy for style.
- **What's a triplet?** (anchor, positive=same class, negative=diff class); one triplet-loss example.
- **Why sample not enumerate?** billions of triplets; resample each epoch for variety.
- **Why are random triplets weak?** many trivially easy (loss ~0) -> motivates hard-negative mining.
- **Limitation of category triplets?** coarse proxy; different-looking same-category items treated similar.

### Common mistakes
- Positive drawn from a different category (inverts signal).
- Anchor == positive image (trivial).
- Only easy random negatives -> weak learning.
- Not resampling -> memorizes a fixed triplet set.
