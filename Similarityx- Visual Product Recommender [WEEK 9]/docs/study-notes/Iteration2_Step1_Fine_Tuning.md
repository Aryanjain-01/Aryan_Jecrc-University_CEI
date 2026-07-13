# SimilarityX AI — Study Notes
## Iteration 2 · Step 1 — Fine-Tuning: Which Layers + the Learning-Rate Trap

> Revision note for college evaluation.

---

### What fine-tuning is
2nd transfer-learning mode (1st = feature extraction, all frozen). Unfreeze SOME pretrained layers and
continue training them on your data. Pretrained weights = starting point; nudge them to specialize.
Questions: which layers, and how gently.

### Which layers to freeze / train (and why)
CNN hierarchy: early = universal edges/textures; deep = task-specific semantics.
- **Freeze early layers** — universal (a shirt's edges = a cat's edges). 3 reasons: already optimal;
  4k images can't train 23M params without overfitting; faster/cheaper.
- **Train final layers only** — high-level semantics; where "ImageNet object parts" -> "fashion
  attributes." The gain lives here.
Rule: **freeze the general bottom, adapt the specialized top.** How many top layers = knob trading
adaptation vs overfitting/compute. Little data -> unfreeze few.

### The learning-rate trap (memorize)
Fine-tune with a VERY LOW learning rate (10-100x smaller than from-scratch). Pretrained weights are
already excellent; a normal/large LR makes huge first updates that OVERWRITE the good features before
they can adapt = **catastrophic forgetting**. Tiny LR = small careful nudges that preserve pretrained
knowledge while steering to fashion. Big LR on pretrained = smashing a finished sculpture; small LR =
fine sanding.
Pro details: keep **BatchNorm frozen / inference mode** during fine-tuning (small-batch stats corrupt
running means/vars); often use **differential LRs** (smaller lower, larger top).

### Our CPU-friendly strategy (defensible engineering choice)
Full-backbone fine-tuning impractical on MacBook Air (23M params, no GPU, 4k images -> slow + overfit).
So: **freeze the whole ResNet50 backbone; train a lightweight projection head** (e.g. 2048->256->128,
L2-normalized) with metric-learning loss. Re-organizes the frozen features into a new space where
"same style" is close. Spirit of fine-tuning (adapt representation) but fast + regularized.

Two wins:
1. **Train on precomputed embeddings.** Backbone frozen -> its 2048-d output never changes -> we already
   saved all 3,964 in embeddings.pkl. Head trains on those VECTORS, no ResNet50 forward passes ->
   **seconds/epoch on CPU** (tiny MLP over 3964x2048).
2. **Regularization by construction** — small head, few params, can't overfit 4k like a 23M backbone.

Eval line: *"With a GPU + more data I'd unfreeze ResNet50's last conv block and fine-tune at low LR.
On CPU with 4k images I freeze the backbone and train a projection head with triplet loss on the
precomputed embeddings — same goal (fashion-specialized space), cheaper, less overfitting."*

---

### Interview Q&A
- **Why freeze early layers?** universal features + avoid overfitting 4k + speed.
- **Why very low LR for fine-tuning?** prevent catastrophic forgetting (big updates overwrite good
  weights before they adapt).
- **What is catastrophic forgetting?** model destroying pretrained knowledge under too-large updates.
- **Why a projection head vs whole backbone here?** CPU + small data; head-on-frozen-features is fast
  (trains on precomputed embeddings), regularizes, still re-organizes the space.
- **BatchNorm gotcha?** keep BN in inference mode during fine-tuning.

### Common mistakes
- Normal LR when fine-tuning -> wrecks pretrained features.
- Unfreeze everything with tiny data -> overfitting.
- Forgetting BN stays inference-mode.
- "More unfrozen = always better" — it's bias/variance tradeoff.
