# SimilarityX AI — Study Notes
## Phase 1 · Step 1 — The Business Problem & Visual Similarity

> Revision note for college evaluation. Read this the night before.

---

### 1. The scenario (the "why")
You are on the search team at Myntra / ASOS / Amazon Fashion. A user sees a product
they like (a screenshot, a camera photo) but **cannot describe it in words**. Your job:
take that **image** and return the 5 most **visually similar** products in the catalog.

Everything we build is machinery to answer one question:
**"Given this image, what do we have that looks like it?"**

---

### 2. Why keyword / text search fails
Traditional search (Elasticsearch, SQL `LIKE`, tag matching) works on **words**, not pixels.

1. **Vocabulary gap** — text search only matches what someone *labeled*. The catalog says
   "blue casual shirt"; the user thinks "navy oxford button-down." Same look, different words.
2. **The query is an image** — in our use case there is no query string at all. A keyword engine
   has nothing to search with.
3. **Visual nuance is unlabelable** — two shirts can share every tag yet look totally different
   (paisley vs. plain). No realistic tag set captures "does it look like this?".

This is the **semantic gap**: the distance between low-level pixels and high-level meaning.

**Reframe:** stop searching over words; search over **appearance**.
This is *visual similarity search* / *content-based image retrieval (CBIR)*.

---

### 3. What "visual similarity" means — core intuition
**Step A — image -> vector (embedding).** Pass each image through a deep network that outputs a
fixed-length vector (2048 numbers for us). Think of it as the image's **coordinates on a map**.
The network is trained so images that *look* alike get *nearby* coordinates.

**Step B — similarity = distance.** Once every image is a point, "find similar" becomes "which
points are closest to my query point?" Compute the query vector, measure distance to every catalog
vector, return the nearest few.

**Memorize this analogy:** it's a **map of a city**. Two restaurants near each other are in the
same neighborhood. The network draws a map where "near" = "looks similar." Recommendation =
"what's near this pin?"

---

### 4. The Iteration 1 pipeline (one breath)
Query image -> **ResNet50 encoder** -> 2048-d embedding -> compare to pre-computed embeddings of all
~4,000 catalog images via **cosine similarity** -> return **Top-5** closest -> show with scores +
metadata.

Iteration 2 improves the **map itself** (re-train the encoder with metric learning so
neighborhoods are tighter and cleaner).

---

### 5. How industry does it (name-drop in the eval)
- **Pinterest Lens** — point camera at object -> embed crop -> nearest-neighbor over billions of pins.
- **Google Lens** — embed query image -> retrieve visually/semantically similar results.
- **Amazon StyleSnap** — upload fashion photo -> shoppable lookalikes via visual embeddings.
- **Myntra / Flipkart** — "similar products" rows + camera search = embedding retrieval over catalog.
- **ASOS Style Match** — snap a photo -> find matching stock.

**Difference from us:** scale (billions -> they need FAISS / vector DBs = our Future Iteration)
and encoder quality (they fine-tune / use CLIP / ViT = our Iteration 2 + Future).
**Core idea is identical.** Good line: *"I built the same system as Pinterest Lens, minus the
scale infrastructure."*

---

### 6. Likely evaluator questions
1. **Why not keyword search?** -> vocabulary gap + query is an image (no words) + semantic gap.
2. **What is an embedding (1 sentence)?** -> a fixed-length vector encoding what an image looks
   like, positioned so similar images are nearby.
3. **How does "find similar" become a computation?** -> nearest-neighbor search; distance in
   embedding space ~ visual dissimilarity.
4. **Is this classification?** -> No. It's **retrieval / ranking** — a ranked list, no fixed output
   classes. Classification is only a *tool we borrow* to get embeddings.
5. **What is the semantic gap?** -> disconnect between raw pixels and high-level meaning; embeddings
   bridge it.

---

### 7. Common mistakes (do NOT say these)
- Calling it "image classification." It's **retrieval**.
- Claiming the model "understands fashion." It learned a **geometry** where visual similarity ~
  proximity.
- Saying similarity is measured "in pixels." No — pixel comparison is brittle. We compare in
  **embedding space**, robust to shifts, lighting, crops.

---

### One-line summary
> We convert every product image into a vector such that look-alike products are close together,
> then answer "find similar" as "find the nearest vectors."
