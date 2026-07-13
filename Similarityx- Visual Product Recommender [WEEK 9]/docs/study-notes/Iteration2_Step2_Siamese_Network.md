# SimilarityX AI — Study Notes
## Iteration 2 · Step 2 — The Siamese Network (Twins & Shared Weights)

> Revision note for college evaluation.

---

### Core idea
A Siamese network is a PATTERN, not a new layer: take ONE encoder and apply it to 2 or 3 inputs,
producing embeddings in the SAME space, then compare them. "Siamese" = twins = identical joined towers
(literally the same network). Solves what a single-input classifier can't: compare multiple inputs.

### Shared weights (the non-negotiable property)
All towers share the EXACT same weights. Why it must be true: if anchor went through encoder-A and
positive through a different encoder-B, "distance between embeddings" is meaningless (two different
maps). Distance is only interpretable if both inputs are mapped by the SAME function into the SAME
space. Shared weights guarantee that + learn one consistent notion of similarity + fewer params.
**In Keras:** weight sharing is automatic if you REUSE the same layer/model object. We built
`projection_head` once and call that same object on A, P, N -> all three use identical weights; a
gradient update improves the one shared head regardless of which tower produced the error. (Do NOT make
3 copies — call 1 object 3 times.)

### Twins vs triplets (2 vs 3 towers)
- **2-tower / pair:** inputs (A, B) + label similar/not. Trained with **contrastive loss**. Simpler.
- **3-tower / triplet:** inputs (anchor, positive, negative) at once. Trained with **triplet loss**.
  We use this — triplet loss is generally stronger (see Step 4).

### Anchor / Positive / Negative
- **Anchor (A)** — reference image (a specific shirt).
- **Positive (P)** — should be CLOSE to anchor (another shirt, same category).
- **Negative (N)** — should be FAR from anchor (a watch, different category).
Loss pulls A&P together, pushes A&N apart. Over thousands of triplets the space reorganizes so
visual/semantic similarity = proximity = the fashion-tuned map.

### Our Step 2 model
Takes 3 inputs (each 2048-d frozen embedding: A, P, N) -> runs all 3 through the ONE shared
`projection_head` -> outputs the 3 resulting 128-d embeddings (stacked) so the **triplet loss (Step 4)**
can compute "pull A-P close, push A-N apart." The Siamese wrapper = plumbing that yields 3 comparable
embeddings from one shared encoder.

---

### Interview Q&A
- **What is a Siamese network?** a pattern: one shared encoder applied to multiple inputs to compare
  their embeddings.
- **Why must towers share weights?** so all inputs map through the same function into the same space ->
  distances are meaningful; also fewer params + one consistent similarity.
- **How is sharing done in Keras?** reuse the same layer/model object (call it N times), don't copy it.
- **Pair vs triplet?** 2-tower + contrastive loss vs 3-tower + triplet loss (we use triplet).
- **Define anchor/positive/negative.** reference / should-be-close (same class) / should-be-far
  (different class).

### Common mistakes
- Building 3 separate encoders (breaks weight sharing -> meaningless distances).
- Thinking Siamese is a special layer — it's a training pattern.
- Confusing the roles: positive = similar to anchor, negative = dissimilar.
