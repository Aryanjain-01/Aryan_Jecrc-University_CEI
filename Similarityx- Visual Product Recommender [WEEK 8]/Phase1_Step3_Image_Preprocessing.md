# SimilarityX AI — Study Notes
## Phase 1 · Step 3 — Image Preprocessing

> Revision note for college evaluation.

---

### Mental model
ResNet50 is a chef trained on ingredients prepared ONE specific way (224x224, seasoned with a
particular blend = ImageNet normalization). Preprocessing prepares each image in the EXACT form
the network saw in training, so its learned filters fire correctly. Get it wrong -> embeddings are
silently garbage (no error, just bad results). That silent failure is why it matters.

---

### 1. Why resize (fixed-size input)?
1. **Batching needs uniform shape.** A batch is one tensor (N, H, W, 3) — only possible if every
   image shares H and W. Ragged sizes can't stack.
2. **Pretrained net expects fixed input.** ResNet50 trained on 224x224; funnels input through 5
   stride-2 stages (224->112->56->28->14->7) to a 7x7 map -> GAP -> 2048 vector. Different size ->
   filters see a mismatched scale.

### 2. Why exactly 224x224?
Convention inherited from ImageNet — the weights were learned at 224x224, so filters operate at the
scale they were tuned for (a collar/bezel spans the expected pixel range). You *can* use other sizes
(ResNet is mostly convolutional) but deviating shifts the real-world scale of receptive fields ->
degraded features. **Rule: match input size to pretraining size.** 224 also halves cleanly 5 times.

### 3. Why normalization (and why NOT just /255)?
Raw pixels are 0-255 ints. Nets train/behave better with inputs centered near 0 on a modest scale
(stable activations & gradients). **Deeper point for transfer learning:** you must reproduce the
EXACT normalization used in training. ResNet50 saw images with the **ImageNet per-channel mean
subtracted** — not [0,1], not [0,255]. So `image/255.0` is the classic WRONG normalization for
ResNet50: it shifts the input distribution and every tuned filter responds wrongly.

### 4. Why ImageNet statistics specifically?
ImageNet per-channel mean (over ~1.2M images) ~ **R=123.68, G=116.78, B=103.94**. Training
subtracted these so data was centered on ImageNet's "average image"; the weights expect that.
Keras `resnet50.preprocess_input` does this in **"caffe" mode**:
  1. RGB -> BGR (ResNet50's original Caffe training used BGR order),
  2. subtract the ImageNet per-channel mean. (No /255, no /std for this model.)

**Key takeaway: never hand-roll normalization — call the model's own `preprocess_input`.** Different
families differ (EfficientNet -> [-1,1]; torch-style uses mean AND std), which is exactly why you
use the matching function.

### 5. Full pipeline
```
raw JPG -> load (Pillow) -> convert RGB -> resize 224x224
        -> float32 array -> resnet50.preprocess_input (RGB->BGR, - ImageNet mean)
        -> add batch dim -> tensor ready
```
Robustness details:
- **Convert to RGB explicitly** — some images are grayscale (1ch) or RGBA (4ch); model needs
  (224,224,3) or it crashes. `.convert("RGB")` fixes all.
- **Aspect-ratio distortion** — square resize squishes tall photos. Alternatives: center-crop or
  pad-to-square. Baseline uses plain resize (fine; ImageNet also cropped) — but name the tradeoff.
- **Don't augment here** — feature extraction wants a deterministic, faithful embedding per catalog
  image. Augmentation is for *training*, not indexing.

---

### Interview Q&A
- **Why 224x224?** matches ResNet50 ImageNet pretraining resolution.
- **Why not /255?** ResNet50 expects ImageNet mean-subtraction in BGR (caffe), not [0,1] -> wrong
  normalization -> distribution shift -> degraded features.
- **What does preprocess_input do (ResNet50)?** RGB->BGR + subtract per-channel ImageNet mean.
- **Grayscale / PNG-with-alpha?** convert to RGB first, else channel count breaks the model.
- **Do you augment here?** No — deterministic embeddings for indexing; augmentation is training-only.

### Common mistakes
- Using /255 or a custom mean with a caffe-mode model.
- Forgetting RGB->BGR (silent quality loss).
- Not handling grayscale/RGBA -> crashes on some catalog images.
- Wrong dtype — must be float32 before preprocess_input.
