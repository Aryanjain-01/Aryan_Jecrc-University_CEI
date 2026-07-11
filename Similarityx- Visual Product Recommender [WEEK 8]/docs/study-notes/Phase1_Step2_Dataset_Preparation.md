# SimilarityX AI — Study Notes
## Phase 1 · Step 2 — Dataset Preparation & the Balanced Subset

> Revision note for college evaluation.

---

### 1. Why subset at all? (5 reasons — know 3)
Full dataset ~44,000 images, ~140 article types. We use a balanced ~4,000.

1. **Compute reality** — we run every image through ResNet50 once to get its embedding.
   ~0.1-0.5s/image on CPU. 44k -> hours per run (and we run many times in 10 days). 4k -> minutes.
2. **Severe class imbalance** — Tshirts ~7,000 vs some types with dozens. Raw -> nearest-neighbor
   biased toward huge classes; metrics inflated & meaningless. Balanced ~500/class fixes this.
3. **Controlled experiment** — the project is a comparison (Baseline vs Fine-tuned vs Siamese).
   To attribute improvement to one change, hold everything else constant -> fixed balanced subset.
4. **Interpretable evaluation** — 8 known categories -> Precision@K per category + real error
   analysis ("confuses Sandals with Shoes — they look alike"). 140 messy types -> hopeless.
5. **Storage & simplicity** — 25GB vs a few hundred MB.

Eval framing: *"Subsetting is a deliberate experimental-design choice for a balanced,
compute-feasible, interpretable benchmark — not a shortcut. Production uses the full catalog + a
vector DB."*

---

### 2. Every column in styles.csv (one row per product)
| Column | What | Why it matters |
|---|---|---|
| id | Integer product ID; image = images/{id}.jpg | **Join key** between metadata & pixels |
| gender | Men/Women/Boys/Girls/Unisex | Display; source of within-class variety |
| masterCategory | Apparel/Footwear/Accessories/Personal Care | Too coarse for a label |
| subCategory | Topwear/Bottomwear/Shoes/Bags/Watches/Sandal | Maps some categories (Bags, Watches) |
| articleType | Shirts/Jeans/Kurtas/Dresses/Casual Shoes/Sandals/Handbags/Watches | **Main filter column** |
| baseColour | Navy Blue/Black/White | Display; also a similarity *confounder* |
| season | Fall/Summer/Winter/Spring | Display only |
| year | 2011, 2012... | Display only |
| usage | Casual/Formal/Sports/Ethnic | Display; explains some clustering |
| productDisplayName | Free-text title | Product name shown to user; source of parse bug |

**Category -> column mapping subtlety:**
- Clean via `articleType`: Shirts, Jeans, Kurtas, Dresses, Sandals, Watches.
- **Bags** -> subCategory=Bags / articleType=Handbags.
- **Shoes** -> NOT one type; spans Casual Shoes / Sports Shoes / Formal Shoes.
=> use an explicit category->filter map; always inspect value distributions before choosing labels.

---

### 3. Famous cleaning problems (mention proactively)
1. **Comma-in-title parse error** — productDisplayName may contain commas -> extra fields ->
   pd.read_csv misaligns/crashes. Fix: `on_bad_lines='skip'`.
2. **Metadata without images (and vice versa)** — not every id has {id}.jpg. Keep only rows whose
   image file actually exists, else crashes at demo time.
3. **Missing values & duplicates** — NaN baseColour/year/usage; dupes. Drop essential-field NaNs,
   de-dup on id.

---

### 4. Balanced-subset algorithm (logic)
1. Load styles.csv safely (skip bad lines).
2. Build category map (8 buckets -> the articleType/subCategory values).
3. Keep rows in the 8 categories AND whose image file exists.
4. Per category: randomly sample 500 with a **fixed random_state** (reproducible = fair
   comparison). If <500 available, take all and note shortfall.
5. Concatenate -> subset.csv (~4,000 rows); copy those images to data/subset/.
6. **Visualize**: bar chart of per-category counts (proves balance) + grid of sample images per
   category (sanity-checks the label map BEFORE it poisons the pipeline).

Internalize: **reproducibility (random_state)** makes the comparison scientifically valid;
**visual sanity check** is a professional habit — never trust an un-eyeballed label column.

---

### 5. How industry handles this
No "download a CSV." Catalog lives in a feature store / warehouse (petabytes), curated by
data-engineering pipelines (Spark, Airflow) that dedupe, validate image availability, and balance
sampling for *training* while serving the *full* catalog at inference. Same three obsessions:
join integrity (id<->image), class balance for training, clean held-out eval sets.

---

### 6. Interview Q&A
- **Why 500/class not everything?** balance + compute + controlled comparison.
- **Which column is your label, why?** mainly articleType (finest clean granularity) + explicit
  map because Shoes/Bags span multiple values.
- **How is Baseline vs Siamese compared fairly?** same fixed subset (random_state); only encoder
  changes.
- **What data-quality issues did you handle?** comma-parse rows, missing image<->metadata joins,
  NaNs/dupes.
- **Risk of imbalanced retrieval set?** large classes dominate neighbors -> inflated metrics.
