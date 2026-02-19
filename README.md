# Adversarial Patch Attacks on DINOv3

Adversarial patch attacks targeting DINOv3 vision transformer for semantic segmentation disruption on Cityscapes driving sequences.

**Compatible Mac M3 Max (MPS) / CUDA / CPU**

## Overview

This project trains adversarial patches that fool a semantic segmentation classifier built on top of DINOv3 (ViT-S/16). The attack makes pedestrians (or other classes) disappear from or be misclassified in the segmentation output.

**Pipeline:**
1. **Train a classifier** — linear probe on DINOv3 tokens, trained on Cityscapes labels
2. **Train an adversarial patch** — optimized to fool the classifier on a target class
3. **Visualize** — sequence visualization showing segmentation before/after attack

---

## Quick Start

```bash
# Install dependencies
uv sync

# Place DINOv3 weights at:
# src/models/weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth

# 1. Train the classifier
python scripts/train_classifier.py

# 2. Train the adversarial patch
python scripts/attack_classifier.py

# 3. Visualize on a sequence
python scripts/visualize_sequence.py --patch results/targeted_patch_final.pt --mode classifier
```

All default parameters are configured in `src/utils/config.py` — no CLI arguments required.

---

## Project Structure

```
.
├── data/                           # Images & datasets
│   ├── stuttgart_00/               # Driving sequence (599 frames)
│   ├── leftImg8bit_trainvaltest/   # Cityscapes images
│   └── gtFine_trainvaltest-2/      # Cityscapes labels
├── results/                        # Output: patches, visualizations
├── scripts/
│   ├── train_classifier.py         # Train linear probe on Cityscapes tokens
│   ├── attack_classifier.py        # Train adversarial patch against classifier
│   ├── visualize_sequence.py       # Visualize attack effect on image sequence
│   └── train_dataset.py            # Universal embedding attack on dataset
├── src/
│   ├── models/
│   │   ├── dinov3/                 # DINOv3 model code
│   │   ├── dinov3_loader.py        # Model loader
│   │   └── weights/                # Model weights (not tracked by git)
│   ├── attacks/
│   │   └── eot_attack.py           # EOT (Expectation Over Transformation)
│   └── utils/
│       ├── config.py               # Centralized default parameters
│       └── viz.py                  # Shared visualization utilities
```

---

## Configuration

All default parameters live in `src/utils/config.py`. Edit this file to change defaults without passing CLI arguments.

```python
# Model
IMG_SIZE = 672          # Input resolution: 224=14×14, 448=28×28, 672=42×42 tokens

# Classifier training
CLF_EPOCHS = 20
CLF_LR = 0.001

# Attack
SOURCE_CLASS = 11       # person
TARGET_CLASS = -1       # -1 = untargeted
ATTACK_STEPS = 2000
ATTACK_LR = 0.05
PATCH_SIZE = 112        # Patch size on image (~17% of IMG_SIZE)
PATCH_RES = 256         # Internal optimization resolution
PATCH_PERSPECTIVE_MIN_SCALE = 0.3  # Perspective scaling (0.3 = 3× smaller at top)

# Visualization
VIZ_SIZE = 500          # Panel size during attack training
VIZ_EVERY = 10          # Live viz every N steps
VIZ_SEQ_SIZE = 300      # Panel size for sequence visualization
```

---

## Step 1 — Train Linear Probe

Trains a `nn.Linear(384, 19)` classifier on DINOv3 token embeddings using Cityscapes ground truth labels.

**Required data** (download from [cityscapes-dataset.com](https://www.cityscapes-dataset.com/downloads/)):
- `leftImg8bit_trainvaltest.zip` → `data/leftImg8bit_trainvaltest/leftImg8bit/train/`
- `gtFine_trainvaltest.zip` → `data/gtFine_trainvaltest-2/gtFine/train/`

```bash
# With all defaults from config.py
python scripts/train_classifier.py

# Override specific params
python scripts/train_classifier.py --img-size 448 --epochs 30
```

A live window shows ground truth vs. predicted segmentation on 4 representative images at each epoch.

| Param | Default | Description |
|-------|---------|-------------|
| `--images` | `data/leftImg8bit.../train` | Cityscapes image folder |
| `--labels` | `data/gtFine.../train` | Cityscapes label folder |
| `--img-size` | 672 | Input resolution |
| `--epochs` | 20 | Training epochs |
| `--lr` | 0.001 | Learning rate |
| `--output` | `results` | Output directory |

**Resolution vs. segmentation quality:**

| `--img-size` | Grid | Tokens | Notes |
|-------------|------|--------|-------|
| 224 | 14×14 | 196 | Fast, coarse |
| 448 | 28×28 | 784 | 4× finer |
| **672** | **42×42** | **1764** | **Default — fine segmentation** |

Output: `results/classifier.pt` (includes `img_size` metadata).

---

## Step 2 — Train Adversarial Patch

Optimizes a patch that causes tokens of the source class to be misclassified. The patch is optimized at high internal resolution (`PATCH_RES`) and bilinearly downsampled to `PATCH_SIZE` before being applied.

**Perspective scaling**: patch size varies with vertical position, simulating a driving camera — smaller at the top (far), larger at the bottom (close).

```bash
# With all defaults (untargeted: person → anything)
python scripts/attack_classifier.py

# Targeted: person → road
python scripts/attack_classifier.py --source-class 11 --target-class 0

# Disable perspective scaling
python scripts/attack_classifier.py --perspective-min-scale 1.0
```

A live window updates every `--viz-every` steps:

```
[ Image+Patch | Seg Original | Seg Attacked | Patch | Legend ]
```

Only focus classes are colored (road, person, car); all others appear gray. Press `q` to stop early.

| Param | Default | Description |
|-------|---------|-------------|
| `--source-class` | 11 (person) | Class to fool |
| `--target-class` | -1 | Target class (-1 = any misclassification) |
| `--steps` | 2000 | Optimization steps |
| `--patch-size` | 112 | Patch size on image (pixels) |
| `--patch-res` | 256 | Internal patch resolution (optimized size) |
| `--perspective-min-scale` | 0.3 | Min scale factor at top of image (1.0 = disabled) |
| `--lr` | 0.05 | Learning rate |
| `--batch-size` | 4 | Images per step |
| `--viz-every` | 10 | Live visualization interval (0 = off) |
| `--viz-size` | 500 | Visualization panel size (px) |
| `--output` | `results` | Output directory |

The `img_size` is read automatically from the classifier checkpoint.

Outputs: `results/targeted_patch_best.pt`, `results/targeted_patch_final.pt`, `results/targeted_attack_results.png`.

**Cityscapes classes:** road(0), sidewalk(1), building(2), wall(3), fence(4), pole(5), traffic_light(6), traffic_sign(7), vegetation(8), terrain(9), sky(10), **person(11)**, rider(12), car(13), truck(14), bus(15), train(16), motorcycle(17), bicycle(18)

---

## Step 3 — Visualize on Sequence

Applies the trained patch to every frame of a driving sequence and visualizes the segmentation effect.

```bash
# Classifier mode (recommended)
python scripts/visualize_sequence.py \
    --patch results/targeted_patch_final.pt \
    --mode classifier \
    --fps 15

# Save as video
python scripts/visualize_sequence.py \
    --patch results/targeted_patch_final.pt \
    --mode classifier \
    --output results/attack_video.mp4
```

**Classifier mode layout:**

```
[ Image+Patch | Seg Original | Seg Attacked | Diff | Legend ]
```

Diff: green = source class unchanged, red = successfully fooled.

**All modes:**

| Mode | Layout | Description |
|------|--------|-------------|
| `classifier` | 1×4 + legend | Semantic segmentation with fooling diff |
| `all` | 2×4 grid | PCA + K-means + trajectory + distance heatmap |
| `pca` | 1×3 | PCA token visualization |
| `segment` | 1×4 | K-means segmentation |
| `trajectory` | 1×2 | Token displacement in PCA space |
| `both` | 1×4 | PCA + distance heatmap |

| Param | Default | Description |
|-------|---------|-------------|
| `--patch` | required | Path to patch file (.pt) |
| `--dataset` | `data/stuttgart_00` | Image folder |
| `--classifier` | `results/classifier.pt` | Classifier checkpoint |
| `--patch-size` | 112 | Patch size (auto-resizes if saved at different size) |
| `--patch-pos` | `50 50` | Patch position (row, col) |
| `--mode` | `all` | Visualization mode |
| `--size` | 300 | Panel size (px) |
| `--fps` | 10 | Display / output video FPS |
| `--source-class` | 11 | Class being attacked |
| `--target-class` | -1 | Target class |
| `--focus-classes` | `0 11 13` | Classes to color (road, person, car) |
| `--smooth` | off | Smooth interpolation |
| `--refresh` | 50 | Refresh PCA/K-means every N frames |
| `--output` | None | Save to MP4 |

**Controls:** `q` quit · `Space` pause/resume

---

## How It Works

### DINOv3 Token Extraction

```
Image (672×672)
     ↓
Split into 42×42 patches (16×16 pixels each)
     ↓
DINOv3 Vision Transformer (ViT-S/16, 384-dim)
     ↓
1764 patch tokens  +  1 CLS token
     ↓
Linear probe: 384 → 19 classes (per token)
     ↓
Segmentation map (42×42 = 1764 "pixels")
```

### Attack Objective

```
For each step:
  1. Sample a batch of images containing source_class
  2. Place patch with perspective-scaled size (smaller = higher = farther)
  3. Extract tokens from patched image
  4. Loss = -CE(adv_logits[source_mask], source_class)   # untargeted
          OR CE(adv_logits[source_mask], target_class)   # targeted
  5. Backprop through classifier + DINOv3 → update patch
  6. Clamp patch ∈ [0, 1]
```

The patch is optimized at `PATCH_RES` resolution and bilinearly downsampled to `PATCH_SIZE` at inference — decoupling optimization quality from physical patch size.

---

## Dataset Structure

```
data/
├── stuttgart_00/                    # Driving sequence for visualization
│   ├── stuttgart_00_000000_000001_leftImg8bit.png
│   └── ... (599 frames)
├── leftImg8bit_trainvaltest/
│   └── leftImg8bit/train/
│       ├── aachen/
│       ├── stuttgart/
│       └── ... (18 cities)
└── gtFine_trainvaltest-2/
    └── gtFine/train/
        ├── aachen/
        │   └── *_gtFine_labelIds.png
        └── ...
```

---

## Development

```bash
# Run tests
pytest

# Format code
black src/ scripts/
ruff check src/ scripts/
```
