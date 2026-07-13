# SimilarityX AI — Study Notes
## Iteration 2 · Step 5 — Training (and Every Hyperparameter)

> Revision note for college evaluation.

---

### What happens during training
Compile the Siamese model with triplet loss + ordering-accuracy metric; feed thousands of (A,P,N)
triplets. Per batch: embed all three via the SHARED head, loss checks "is N farther than P by the
margin?", backprop nudges ONLY the head's weights (ResNet50 stays frozen). Over epochs the space
reshapes: same-category clusters, different-category separates.

### Hyperparameters (know each)
- **Margin (0.2)** — required gap between d(A,P) and d(A,N). Too small -> weak separation; too large
  -> unsatisfiable/unstable. 0.2 sensible for squared distances on unit sphere (range 0-4).
- **Learning rate (1e-3, normal Adam)** — SUBTLETY vs Step 1: the tiny-LR rule was for PRETRAINED
  weights we must not destroy. Our head is RANDOMLY INITIALIZED (nothing precious) -> a normal LR
  trains it efficiently. The frozen ResNet50 (precious pretrained weights) isn't trained at all.
  => tiny LR for pretrained layers being adapted; normal LR for a fresh head from scratch.
- **Optimizer (Adam)** — adaptive, robust default.
- **Epochs (~30)** — one pass over sampled triplets; watch ordering-accuracy plateau near ~1.0. Tiny
  head on precomputed vectors -> seconds/epoch.
- **Batch size (128)** — triplets per gradient step; larger = smoother + faster on the vectorized head.
- **#triplets (~20,000) + val split 10%** — large varied set; held-out val_triplet_accuracy warns of
  overfitting.

### What to watch
- loss falls toward ~0 (most triplets satisfy margin).
- triplet_accuracy climbs toward ~1.0 (N correctly farther than P).
- train and val track each other; big gap = overfitting.

**Honest caveat:** labels are CATEGORY-level (coarse proxy) -> high ordering-accuracy is easy and is
NOT the real proof. Real proof = retrieval quality in Step 8 (Precision@K on actual recommendations).

### After training
Save the trained projection head to artifacts/models/. Step 6 composes frozen ResNet50 -> trained head
to produce IMPROVED embeddings for the whole catalog.

---

### Interview Q&A
- **Why a normal LR here vs tiny for fine-tuning?** head is randomly initialized (train from scratch);
  tiny-LR rule is for pretrained weights, which stay frozen here.
- **What does the margin control; too big/small?** required P-N gap; small=weak separation, big=
  unsatisfiable/unstable.
- **How do you know training works?** loss down, ordering-accuracy up toward 1.0, train/val track.
- **Why isn't high training accuracy enough?** category labels are coarse; real proof = Step 8 retrieval.

### Common mistakes
- Tiny fine-tuning LR for a fresh head -> painfully slow.
- Judging success by training accuracy alone vs held-out retrieval quality.
- Margin mismatched to distance scale (margin 5 with squared distances capped at 4 -> never satisfiable).
