# SimilarityX AI — Study Notes
## Iteration 2 · Step 4 — Loss Functions (Contrastive, Triplet, Hard Mining)

> Revision note for college evaluation. The intellectual core of Iteration 2.

The loss IS the training signal — the math statement of "what is a good embedding space?"
Setup: embeddings are L2-normalized (unit sphere), so distance d(x,y)=||x-y|| (monotone with cosine:
smaller distance = higher cosine).

---

### Contrastive loss (pairs)
Pair (x1,x2), label Y=0 similar / Y=1 dissimilar, margin m:
```
L = (1-Y) * 0.5 * d^2   +   Y * 0.5 * [max(0, m - d)]^2
```
- **Similar (Y=0):** L = 0.5 d^2 -> minimized by d->0 -> PULL together.
- **Dissimilar (Y=1):** L = 0.5 max(0, m-d)^2 -> positive only while d<m; once >= m apart, loss 0 ->
  PUSH apart but only up to the margin.
Geometry: similar collapse to a point; dissimilar pushed outside a radius-m bubble.
**Weakness:** ABSOLUTE distance targets (similar~0, dissimilar>=m). No single correct absolute distance;
forcing all positives to exactly 0 is rigid.

### Triplet loss (triplets) — OUR CHOICE
(A,P,N), margin m:
```
L = max(0,  d(A,P) - d(A,N) + m)
```
Demands **d(A,N) > d(A,P) + m** (negative farther than positive by a margin).
- Already satisfied -> loss 0.
- Violated -> gradient pulls P toward A and pushes N away.
**Key difference / why triplet wins:** RELATIVE, not absolute. Never says "positive at distance 0";
only "negative farther than positive by a margin" — cares about ORDERING not exact distances -> more
flexible -> better embeddings. Also uses CONTEXT (P and N jointly per anchor).

### Hard-negative mining (makes triplet training work)
Random triplets are mostly EASY (random shirt vs watch already far -> loss 0 -> no gradient). Feed
INFORMATIVE triplets instead. By the negative's distance:
- **Easy:** d(A,N) > d(A,P)+m -> loss 0. Useless.
- **Semi-hard:** d(A,P) < d(A,N) < d(A,P)+m -> farther than positive but inside margin. Informative +
  STABLE. (FaceNet used semi-hard.)
- **Hard:** d(A,N) < d(A,P) -> negative CLOSER than positive. Most informative but UNSTABLE (noisy).
**Mining** = deliberately pick hard/semi-hard negatives so every triplet gives a useful gradient.
**Batch-hard:** within a batch, per anchor pick hardest positive + hardest negative.
Our setup (frozen embeddings): per anchor sample candidate negatives, keep the closest in current head
space = informative negative (wired in Step 5). Standard triplet loss already ignores easy triplets (0).

### Contrastive vs Triplet (table)
| | Contrastive | Triplet |
|---|---|---|
| Input | pair + label | triplet (A,P,N) |
| Constraint | ABSOLUTE distances | RELATIVE ordering |
| P & N together? | no | YES (per anchor) |
| Result | good but rigid | usually better |

### Why triplet preferred (one-liner)
*"Contrastive pins similar to 0 and dissimilar outside a fixed radius (absolute, hard to satisfy
globally). Triplet only requires the negative be farther than the positive by a margin (relative
ranking) — more flexible, and with hard-negative mining learns stronger embeddings. FaceNet used it."*

---

### Interview Q&A
- **Write triplet loss.** max(0, d(A,P) - d(A,N) + margin).
- **What does the margin do?** forces a gap (N farther than P by >= m); prevents trivial collapse
  (all distances 0).
- **Contrastive vs triplet core diff?** absolute distance targets vs relative ranking; triplet joins P&N.
- **Hard-negative mining & why?** pick margin-violating (hard/semi-hard) negatives; random ones are
  mostly easy -> zero gradient.
- **Easy vs semi-hard vs hard?** by d(A,N) vs d(A,P) and margin; semi-hard = informative + stable.

### Common mistakes
- No margin -> network collapses everything to one point (all distances 0).
- Random (easy) triplets -> loss stalls, barely learns.
- Hardest negatives from step 1 -> unstable; start semi-hard.
- Un-normalized embeddings -> distance dominated by magnitude not direction.
