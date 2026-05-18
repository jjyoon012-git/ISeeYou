# ISEEYOU N=7 ensemble — handoff for model integration

A 7-model ensemble of EfficientNet-B0-based classifiers for short-video real vs AI-generated detection. This file describes everything an integrator needs: model architecture, the seven checkpoints, preprocessing, the TTA recipe, the three aggregation methods, and recommended operating modes.

---

## 1. Inputs and outputs

- **Input**: a video (any length). The pipeline samples N frames, runs each through 7 models, aggregates within and across frames.
- **Output**: per-video binary classification:
  - `class 0 = real`
  - `class 1 = generated (AI)`
- The model produces a 2-d softmax probability `[p_real, p_gen]`. The default decision rule is `verdict = generated if p_gen > 0.5 else real`. Threshold can be tuned per deployment.

---

## 2. Model architecture (identical for all 7 except `image_size`)

All seven checkpoints share the exact same network definition (from `iseeyou/models/builder.py`):

```python
import timm
import torch.nn as nn

class FrameClassifier(nn.Module):
    def __init__(self, backbone="efficientnet_b0", num_classes=2,
                 pretrained=False, dropout=0.4, hidden_dim=0):
        super().__init__()
        self.backbone = timm.create_model(
            backbone, pretrained=pretrained, num_classes=0, global_pool="avg",
        )
        feature_dim = self.backbone.num_features  # 1280 for efficientnet_b0
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, num_classes),
        )

    def forward(self, x):
        return self.head(self.backbone(x))
```

All seven models use:
- `backbone = "efficientnet_b0"` (timm), `pretrained=False` at load time, weights come from the checkpoint
- `num_classes = 2`
- `dropout = 0.4`
- `hidden_dim = 0` (no hidden FC layer; just Dropout → Linear)

Each `.pt` checkpoint is a dict with at least:
- `model_state_dict` — the weights to `model.load_state_dict(...)` into the network above
  - **Important**: for the six EMA models, this dict is the **EMA snapshot**, not the raw training weights. You don't need to do anything special; just load it.
- `training_cfg` — original training hyperparameters, useful for traceability
- `epoch`, `best_metric`, `best_epoch` — bookkeeping

---

## 3. The seven checkpoints

Files are at: `outputs/checkpoints_<NAME>_frame/best.pt` (each ~46 MB).

| # | label | NAME (`outputs/checkpoints_<NAME>_frame/best.pt`) | image_size | seed | train data | val EMA F1 |
|--:|---|---|:-:|:-:|---|:-:|
| 1 | `robustaug`              | `protocol_youtube_dataset_plus_local_videoonly_clean_robustaug` | 224 | 42 | full clean manifest, no EMA | 0.7788 (raw) |
| 2 | `robustaug+EMA`          | `protocol_youtube_dataset_plus_local_videoonly_clean_robustaug_ema` | 224 | 42 | full clean manifest | 0.7634 |
| 3 | `ff2f_holdout`           | `protocol_youtube_dataset_plus_local_videoonly_clean_robustaug_ema_ff2f_holdout` | 224 | 42 | clean − Face2Face (500) | 0.7566 |
| 4 | `seed1337`               | `protocol_youtube_dataset_plus_local_videoonly_clean_robustaug_ema_seed1337` | 224 | 1337 | full clean manifest | 0.7608 |
| 5 | `seed7`                  | `protocol_youtube_dataset_plus_local_videoonly_clean_robustaug_ema_seed7` | 224 | 7 | full clean manifest | 0.7596 |
| 6 | `df_holdout`             | `protocol_youtube_dataset_plus_local_videoonly_clean_robustaug_ema_df_holdout` | 224 | 42 | clean − Deepfakes (1000) | 0.7408 |
| 7 | `img320`                 | `protocol_youtube_dataset_plus_local_videoonly_clean_robustaug_ema_img320` | **320** | 42 | full clean manifest | 0.7953 |

Diversity:
- **3 random seeds** (42, 1337, 7) — robustaug+EMA seeds 42/1337/7 are identical recipe except for seed
- **2 cross-method holdouts** (Face2Face removed; Deepfakes removed) — these models never saw a particular manipulation method during training
- **1 different input scale** (320 vs 224)
- 1 model without EMA (`robustaug`) for "raw weights" diversity

The diversity is what gives the ensemble its lift; using only seeds 42/42/42 would be much weaker.

Trainer source: `iseeyou/engine/trainer.py` (the EMA logic is the `ModelEMA` class; EMA decay = 0.999 throughout; on save, `best.pt`'s `model_state_dict` is the EMA snapshot).

---

## 4. Preprocessing (apply to every frame before passing to any model)

Order matters. For each sampled frame (an `H×W×3` uint8 RGB numpy array):

### 4.1. Text mask (top + bottom band fill with median color)

Top 8% and bottom 18% of the frame are blanked out by replacing those rows with the median pixel value. This kills subtitle/caption shortcuts.

```python
def apply_text_mask(rgb_np):
    h, w = rgb_np.shape[:2]
    out = rgb_np.copy()
    top = int(round(h * 0.08))
    bot = int(round(h * 0.18))
    if top > 0:
        median_top = np.median(out[:top].reshape(-1, 3), axis=0).astype(np.uint8)
        out[:top] = median_top
    if bot > 0:
        median_bot = np.median(out[-bot:].reshape(-1, 3), axis=0).astype(np.uint8)
        out[-bot:] = median_bot
    return out
```

Apply to every frame. Same masked frame is fed to all 7 models.

### 4.2. Per-model image transform

Each model has its own `image_size`. For models 1–6 use 224; for model 7 (`img320`) use 320.

```python
from torchvision import transforms
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

def make_transform(image_size):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),  # square resize, NOT a center crop
        transforms.ToTensor(),                        # uint8 [0,255] → float32 [0,1]
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
```

Input to the transform is a `PIL.Image.fromarray(masked_rgb)` of dtype uint8. Output is `[3, image_size, image_size]` float32.

### 4.3. Frame sampling

Sample 6 evenly-spaced frames per video:

```python
import cv2, numpy as np
def sample_frames(video_path, n_frames=6):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, total - 1, n_frames).astype(int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, bgr = cap.read()
        if ret:
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames  # list of HxWx3 uint8 RGB
```

(Training used 1 anchor frame per video at `frame_sampling_mode: anchor`, but at inference time 6 frames is more robust and consistent with how the published metrics were measured.)

---

## 5. Test-time augmentation (TTA): horizontal flip

For every batch, run the model **twice** — once on the original tensor, once on the horizontally-flipped tensor — and average the softmax probs.

```python
def predict_with_tta_hflip(model, batch_tensor, device):
    model.eval()
    with torch.no_grad():
        x = batch_tensor.to(device, non_blocking=True)
        p1 = torch.softmax(model(x), dim=1)
        p2 = torch.softmax(model(torch.flip(x, dims=[-1])), dim=1)
    return ((p1 + p2) * 0.5).cpu().numpy()  # [B, 2]
```

This is mandatory — all reported numbers use TTA hflip. Doubles inference cost but is the cheapest single accuracy gain we have.

---

## 6. Aggregation (combining the 7 model outputs into one verdict)

For each frame, you get 7 `[p_real, p_gen]` softmax vectors stacked into a `[7, 2]` array. The 7 are then aggregated into a single `[2]` per-frame probability. There are three supported aggregation methods.

Notation: `P` is a `[7, 2]` array. Output is `[2]`.

### 6.1. mean (arithmetic average)
```python
out = P.mean(axis=0)
```
- Best for: F1, TPR@FPR=1%, ECE.
- Default if you want one mode.

### 6.2. geomean (logit-space average)
```python
log_p = np.log(np.clip(P, 1e-9, 1.0)).mean(axis=0)  # [2]
out = np.exp(log_p)
out = out / out.sum()                                # renormalize
```
- Best for: AUC, TPR@FPR=0.1% (strict false-positive operating point).
- More punishing of any single model that says "low p" — drives ensemble toward agreement.

### 6.3. median
```python
m = np.median(P, axis=0)   # [2]
out = m / m.sum()          # renormalize
```
- Best for: F1 and AUC at N≥6. Robust to single-model outliers. **This is the recommended default for production at N=7.**

### 6.4. (optional) trimmed_mean
```python
sorted_idx = np.argsort(P[:, 1], axis=0)  # sort by p_gen
keep = sorted_idx[1:-1]                   # drop top and bottom
out = P[keep].mean(axis=0)
```
- Drops the highest and lowest p_gen across the 7 models, averages the middle 5. Good for TPR@FPR=0.1% at N=6+.

After the per-frame aggregation, you have a per-frame `[p_real, p_gen]`. Combine across frames with `confidence_mean`:

```python
def confidence_mean_across_frames(probs_2d):  # [n_frames, 2]
    max_p = probs_2d.max(axis=1)
    weights = max_p ** 2
    weights = weights / max(weights.sum(), 1e-9)
    return (probs_2d * weights[:, None]).sum(axis=0)  # [2]
```

This weights each frame by `(max class probability)^2` so confident frames count more than uncertain ones. This is what training's `confidence_mean` aggregator does.

---

## 7. Recommended operating modes

Depending on what the deployment optimizes for, pick one:

| mode | aggregation | image_sizes | best for | test metric |
|---|---|---|---|---|
| **Default / general purpose** | median, no calibration | mix (use all 7) | F1, AUC | F1 0.7805, AUC 0.8588 |
| **Strict TPR@FPR=1%** | median | 224 only (drop img320) | maximizing TPR at 1% FPR | TPR@FPR=1% 0.3130 |
| **Best-calibrated probabilities** | geomean + per-model temperature scaling | 224 only | minimum ECE | ECE 0.0392 |

For most users, **default mode** is the right choice. The other two are niche.

If you want to use temperature scaling (mode 3), each model's logits are scaled by a per-model temperature before softmax:

```python
T = {1:1.6895, 2:1.6352, 3:1.3755, 4:1.5219, 5:1.6840, 6:1.6085, 7:1.5219}
# then for model i:
probs_calibrated = softmax(logits / T[i])
```

These T values were fit by LBFGS minimizing NLL on the val split. T > 1 means each model was over-confident.

---

## 8. Reference performance numbers (frame-level test set, TTA hflip, N=7 default)

| metric | starting baseline (robustaug single, no TTA) | N=7 default | improvement |
|---|---|---|---|
| F1            | 0.7547 | **0.7805** | +3.4% |
| AUC           | 0.8299 | **0.8588** | +3.5% |
| TPR @ FPR=1%  | 0.2112 | 0.3130*    | +48% rel |
| TPR @ FPR=0.1% | 0.0865 | **0.2570** | +197% rel (≈3×) |
| ECE           | 0.0993 | 0.0466     | −53% |

\* TPR@FPR=1%'s peak (0.3130) is N=6 median; N=7 median is 0.2977.

Cross-method generalization (held-out methods the ensemble never fully saw):

| eval set | what's held out | N=7 best AUC | TPR@FPR=1% |
|---|---|---|---|
| ff_holdout (Face2Face) | model 3 never saw it | **0.9920** (mean) | 0.764 (median) |
| df_holdout (Deepfakes) | model 6 never saw it | **0.9913** (trimmed_mean) | 0.726 (median) |

Real-world spot check (12 confirmed AI-generated YouTube Shorts from two channels + 5 confirmed real shorts from a normal channel):

| | predicted REAL | predicted GENERATED |
|---|---|---|
| **actual REAL** (5) | 5 | 0 |
| **actual GENERATED** (12) | 2 | 10 |

→ accuracy 88%, precision 100%, recall 83%, FPR 0%.

---

## 9. End-to-end inference example (single video, N=7 default mode)

```python
from pathlib import Path
import cv2, numpy as np, torch, timm
from PIL import Image
from torchvision import transforms

# 7 model bundle: list of (name, image_size, ckpt_path)
MODELS = [
    ("robustaug",     224, "outputs/checkpoints_protocol_youtube_dataset_plus_local_videoonly_clean_robustaug_frame/best.pt"),
    ("ema",           224, "outputs/checkpoints_protocol_youtube_dataset_plus_local_videoonly_clean_robustaug_ema_frame/best.pt"),
    ("ff2f_holdout",  224, "outputs/checkpoints_protocol_youtube_dataset_plus_local_videoonly_clean_robustaug_ema_ff2f_holdout_frame/best.pt"),
    ("seed1337",      224, "outputs/checkpoints_protocol_youtube_dataset_plus_local_videoonly_clean_robustaug_ema_seed1337_frame/best.pt"),
    ("seed7",         224, "outputs/checkpoints_protocol_youtube_dataset_plus_local_videoonly_clean_robustaug_ema_seed7_frame/best.pt"),
    ("df_holdout",    224, "outputs/checkpoints_protocol_youtube_dataset_plus_local_videoonly_clean_robustaug_ema_df_holdout_frame/best.pt"),
    ("img320",        320, "outputs/checkpoints_protocol_youtube_dataset_plus_local_videoonly_clean_robustaug_ema_img320_frame/best.pt"),
]
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else
                      "cuda" if torch.cuda.is_available() else "cpu")

def build_model():
    backbone = timm.create_model("efficientnet_b0", pretrained=False, num_classes=0, global_pool="avg")
    feature_dim = backbone.num_features
    head = torch.nn.Sequential(torch.nn.Dropout(0.4), torch.nn.Linear(feature_dim, 2))
    return torch.nn.Sequential(  # NB: real implementation uses FrameClassifier class above; sequential shown for illustration
        backbone, head,
    )

def load_models():
    out = []
    for name, size, path in MODELS:
        m = build_model().to(DEVICE)
        ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
        # state_dict is keyed under "backbone.*" and "head.*" — match the FrameClassifier class
        m.load_state_dict(ckpt["model_state_dict"])
        m.eval()
        out.append((name, size, m))
    return out

def make_tfm(size):
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

def text_mask(rgb):
    h = rgb.shape[0]
    out = rgb.copy()
    top = int(round(h * 0.08))
    bot = int(round(h * 0.18))
    if top > 0: out[:top]  = np.median(out[:top].reshape(-1, 3),  axis=0).astype(np.uint8)
    if bot > 0: out[-bot:] = np.median(out[-bot:].reshape(-1, 3), axis=0).astype(np.uint8)
    return out

def sample_frames(video_path, n=6):
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idx = np.linspace(0, total - 1, n).astype(int)
    frames = []
    for i in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ret, bgr = cap.read()
        if ret: frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames

def fwd_tta(model, batch):
    with torch.no_grad():
        x = batch.to(DEVICE, non_blocking=True)
        p = torch.softmax(model(x), dim=1)
        p = (p + torch.softmax(model(torch.flip(x, dims=[-1])), dim=1)) * 0.5
    return p.cpu().numpy()

def predict_video(video_path, models):
    frames = sample_frames(video_path, n=6)
    masked = [text_mask(f) for f in frames]
    tfms = {size: make_tfm(size) for _, size, _ in models}
    n_models = len(models)
    n_frames = len(masked)
    P = np.zeros((n_models, n_frames, 2))
    for i, (name, size, m) in enumerate(models):
        batch = torch.stack([tfms[size](Image.fromarray(f)) for f in masked], dim=0)
        P[i] = fwd_tta(m, batch)

    # Aggregate across models per frame: median (recommended default for N=7)
    median_per_frame = np.median(P, axis=0)
    median_per_frame /= median_per_frame.sum(axis=1, keepdims=True)

    # Aggregate across frames: confidence_mean
    max_p = median_per_frame.max(axis=1)
    w = max_p ** 2
    w /= max(w.sum(), 1e-9)
    final = (median_per_frame * w[:, None]).sum(axis=0)
    return {"p_real": float(final[0]), "p_gen": float(final[1]),
            "verdict": "GENERATED" if final[1] > 0.5 else "REAL"}
```

For the actual production-quality `FrameClassifier` definition, copy `iseeyou/models/builder.py` from this repo so state_dict keys match exactly.

---

## 10. What NOT to do

1. **Don't skip the text mask.** Without it, the model leans on subtitles/captions and fails on real video without text overlay.
2. **Don't change normalization.** Use ImageNet `(0.485, 0.456, 0.406) / (0.229, 0.224, 0.225)`.
3. **Don't crop or letterbox.** Use a square resize directly. The text mask is computed in the original aspect ratio and that's preserved by Resize-to-square.
4. **Don't drop `img320`** unless you're using "Strict TPR@FPR=1%" mode. It contributes both as the best single model and as resolution diversity.
5. **Don't try to ensemble logits across different image_sizes naively** — different image sizes mean different effective receptive fields, so do the per-model softmax first and then aggregate at the probability level (as shown).
6. **Don't load this on a single CPU thread for production**. With 7 models × 2 (TTA) × 6 frames = 84 forwards per video, you'll want batched MPS or CUDA inference. On MPS it's ~2–3 sec per video.

---

## 11. Per-checkpoint sanity checks

For each `best.pt`, you can verify it loaded correctly by running on a known reference video and matching the expected probability range. The ranges below are from real-world spot tests (high-confidence cases):

- High-confidence real video: each model should output `p_gen` in approximately `[0.05, 0.30]`.
- High-confidence generated (commercial AI shorts): each model should output `p_gen` in approximately `[0.80, 0.99]`.
- Out-of-the-six (`robustaug` is the no-EMA model) opinions can disagree by 5–20 percentage points on uncertain inputs — that's normal and is exactly why we ensemble.

---

## 12. Files an integrator needs

Send these files together:

```
outputs/checkpoints_protocol_youtube_dataset_plus_local_videoonly_clean_robustaug_frame/best.pt
outputs/checkpoints_protocol_youtube_dataset_plus_local_videoonly_clean_robustaug_ema_frame/best.pt
outputs/checkpoints_protocol_youtube_dataset_plus_local_videoonly_clean_robustaug_ema_ff2f_holdout_frame/best.pt
outputs/checkpoints_protocol_youtube_dataset_plus_local_videoonly_clean_robustaug_ema_seed1337_frame/best.pt
outputs/checkpoints_protocol_youtube_dataset_plus_local_videoonly_clean_robustaug_ema_seed7_frame/best.pt
outputs/checkpoints_protocol_youtube_dataset_plus_local_videoonly_clean_robustaug_ema_df_holdout_frame/best.pt
outputs/checkpoints_protocol_youtube_dataset_plus_local_videoonly_clean_robustaug_ema_img320_frame/best.pt
iseeyou/models/builder.py               # the FrameClassifier definition
iseeyou/utils/masking.py                # the text-mask implementation (apply_band_mask_np / apply_text_mask_np)
MODEL_HANDOFF.md                        # this file
```

Total ~322 MB for the seven `.pt` files. The two Python files together are <10 KB. Together that's everything.

If they want a single deployable Python file with no `iseeyou/` import, the `FrameClassifier` class and the `apply_text_mask` function are tiny enough to inline directly into their inference script — see Section 9 for a near-complete example.

---

## 13. Contact / source

Code: https://github.com/jeehun3020/ISEEYOU_MODEL  
Detailed phase-by-phase progression (with all eval JSONs and intermediate decisions): `outputs/protocol/youtube_dataset_plus_local_videoonly_clean/phase_{a..f}_*.md`.
