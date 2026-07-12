# SimilarityX AI — Study Notes
## Iteration 2 · Intro — Why Metric Learning & Siamese Networks

> Revision note for college evaluation. The "why Iteration 2 exists" story.

---

### The limitation we're fixing
Baseline embeddings come from ResNet50 trained to classify **ImageNet** (cats, cars). That space was
organized to separate those 1000 classes — NOT to place fashion products so "similar style" = "close."
Cosine then measures distance in a space built for the WRONG objective.
Observed: a shirt query matched other shirts BUT also on model/pose/white background, because ImageNet
features encode "person in a photo" as strongly as "striped shirt." The space doesn't know garment =
signal, background = noise.
**No metric can fix a poorly-organized space — re-organize the space itself.** = Iteration 2.

### Why fine-tuning improves embeddings
Gentlest fix: unfreeze ResNet50's TOP layers, keep training on FASHION data. Early layers =
universal edges/textures (keep frozen). Deep layers = task-specific semantics -> retrain them so they
drift from "ImageNet object parts" toward "fashion attributes." Tightens clusters.
BUT fine-tuning for *classification* still optimizes the wrong thing (predict a label, not make
similar items close). We want to optimize the geometry DIRECTLY -> metric learning.

### Why metric learning exists
A family of techniques whose objective is literally: **learn an embedding space where distance =
semantic similarity.** Training signal = "pull similar together, push dissimilar apart" (not "predict
the class"). For retrieval/recommendation that's exactly our objective, so we optimize it head-on
instead of hoping classification yields good geometry as a side effect.
Analogy: classification arranges a library by fixed genre labels; metric learning arranges it so any
two books you'd call similar end up on neighboring shelves.

### Why Siamese networks were invented
To train "make similar things close" you must feed TWO/THREE images at once and compare their
embeddings — encoded by the SAME function so distances are meaningful. A single-input classifier can't.
**Siamese network** = two/three identical copies of the encoder that **share the exact same weights**
("twins"); each processes one input -> compare embeddings. Shared weights => A and B map through the
same function into the same space => distance is meaningful. Trained so matching pair = small distance,
non-matching = large. Invented for signature verification (1990s); standard tool for learning similarity.

### Industry uses
- **Face recognition** (FaceNet, Apple Face ID) — embed faces; same person close, different far;
  verify by distance. Trained with triplet loss. (Canonical example.)
- **Visual search / shop-the-look** — Pinterest, Amazon, ASOS metric-learn catalog encoders so
  style-matches beat background-matches (our exact problem).
- **Signature/handwriting verification** — original Siamese use case.
- **Speaker verification, de-duplication, recommendation** — any "are these two the same/similar?".

Eval line: *"Iteration 1 borrowed a space built for ImageNet classification. Iteration 2 builds a
space purpose-made for fashion similarity via metric learning — the approach behind FaceNet and
Pinterest visual search."*

### The plan (Steps)
1. Fine-tune encoder's top layers on fashion.
2. Wrap in Siamese architecture (shared weights).
3. Build pairs/triplets: (anchor, positive, negative).
4. Train with metric-learning loss (contrastive / triplet + hard-negative mining).
5-8. Train -> re-embed -> retrieve -> measure improvement (Precision@K / Recall@K vs baseline).

---

### Interview Q&A
- **Why isn't baseline enough?** space organized for ImageNet classes, not fashion similarity; cosine
  measures wrong geometry (matches pose/background).
- **Fine-tuning vs metric learning?** fine-tuning still optimizes classification; metric learning
  optimizes distances directly (what we want).
- **What is a Siamese net & why shared weights?** twin encoders, identical shared weights, so both
  inputs map through the same function into the same space -> meaningful distance.
- **Real system using this?** FaceNet / Face ID (triplet loss); Pinterest/Amazon visual search.
