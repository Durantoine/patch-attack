# Adversarial Patch Attacks on DINOv3

Adversarial patch attacks targeting DINOv3 foundation model for embedding manipulation and segmentation disruption.

**Compatible Mac M3 Max (MPS) / CUDA / CPU**

## Overview

This project implements adversarial patch attacks on Meta's DINOv3 vision transformer model. The attacks target:

1. **Embedding Space Attacks**: Maximize L2 distance or modify embedding norms
2. **Segmentation Coherence Attacks**: Destroy spatial coherence in unsupervised segmentation
3. **Real-time Video Attacks**: Interactive visualization and video processing

## Quick Start

```bash
# Install dependencies
uv sync

# Download DINOv3 weights
# Place checkpoint at: src/models/weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth
```

## Project Structure

```
.
├── data/                    # Images & datasets for training/testing
├── docs/                    # Documentation
├── examples/                # Demo scripts
│   ├── webcam_segmentation.py      # Real-time segmentation visualization
│   ├── interactive_attack.py       # Interactive patch attack with mouse control
│   ├── eot_interactive.py          # EOT attack with 3D orientation control
│   ├── video_demo.py               # Video processing examples
│   ├── demo_dinov3.py              # Basic embedding attack
│   └── demo_segmentation_attack.py # Segmentation coherence attack
├── results/                 # Output visualizations & saved patches
├── scripts/                 # Training & visualization scripts
│   ├── train_patch.py              # Single image/webcam patch training
│   ├── train_dataset.py            # Dataset-based universal patch training
│   └── visualize_sequence.py       # Sequence visualization (video) of attack effects
├── src/                     # Source code
│   ├── attacks/             # Attack implementations
│   │   ├── patch_attack.py         # Base patch attack
│   │   ├── video_attack.py         # Video adversarial attack
│   │   └── eot_attack.py           # EOT (Expectation Over Transformation)
│   ├── models/              # Model loading
│   │   ├── dinov3/                 # DINOv3 model code
│   │   ├── dinov3_loader.py        # Model loader
│   │   └── weights/                # Model weights
│   ├── video_processor.py   # Video feature extraction
│   └── segmentation/        # Segmentation modules
└── tests/                   # Tests
```

---

## Real-time Demos

### 1. Webcam Segmentation

Visualize DINOv3 segmentation in real-time on your webcam.

```bash
python examples/webcam_segmentation.py --size 512
```

**Modes:**
- `pca`: RGB visualization from PCA projection of tokens
- `segment`: K-means clustering colored by cluster
- `both`: Side-by-side comparison

**Controls:**
| Key | Action |
|-----|--------|
| `m` | Change mode (pca/segment/both) |
| `s` | Toggle smooth/pixelated |
| `+/-` | Change number of clusters (2-8) |
| `q` | Quit |

---

### 2. Interactive Attack

Optimize and move an adversarial patch in real-time. See how it affects embeddings.

```bash
# Optimize on current frame (best quality - recommended)
python examples/interactive_attack.py --single-frame --size 500

# Optimize a universal patch on 30 frames
python examples/interactive_attack.py --size 500

# Load a pre-trained patch
python examples/interactive_attack.py --load-patch results/universal_patch.pt --size 500
```

> **Tip**: Use `--single-frame` for a more effective attack (1000 steps, LR=0.05) vs universal mode which optimizes on multiple frames.

**What you see:**

```
┌─────────────────┬─────────────────┬─────────────────┐
│  Image + Patch  │  Reference      │  Attacked       │
│  (drag to move) │  (no patch)     │  (with patch)   │
└─────────────────┴─────────────────┴─────────────────┘
```

**Modes:**
- `pca`: PCA RGB before/after
- `segment`: K-means clusters before/after
- `distance`: Heatmap of token perturbation
- `both`: Full 2x3 grid with all visualizations

**Controls:**
| Key | Action |
|-----|--------|
| **Mouse drag** | Move the patch |
| `m` | Change mode |
| `s` | Toggle smooth |
| `p` | Toggle patch on/off |
| `+/-` | Change clusters |
| `o` | Re-optimize patch |
| `w` | Save current patch |
| `r` | Reset position |
| `q` | Quit |

---

### 3. EOT Interactive (Orientation Variation)

Test patch effectiveness with 3D orientation changes. Simulates a real printed patch viewed from different angles.

```bash
# Interactive orientation control
python examples/eot_interactive.py --size 500

# Load a pre-trained EOT-robust patch
python examples/eot_interactive.py --load-patch results/eot_patch.pt
```

**Controls:**
| Key | Action |
|-----|--------|
| `←/→` | Rotate patch left/right (yaw) |
| `↑/↓` | Tilt patch forward/back (pitch) |
| `[/]` | Roll patch |
| `+/-` | Scale patch |
| `r` | Reset orientation |
| `t` | Train EOT-robust patch |
| `q` | Quit |

**What is EOT?**

EOT (Expectation Over Transformation) trains patches that remain effective even when:
- Viewed from different angles (rotation, perspective)
- At different distances (scale changes)
- Under different lighting (brightness, contrast)

```
Standard patch:    Works only front-on
EOT patch:         Works from multiple angles ✓
```

---

## Patch Training

### Train on Single Image / Webcam (No orientation variation)

Simple training at a fixed position. Fast but not robust to viewing angle changes.

```bash
# Train on a single image
python scripts/train_patch.py --image data/0.png --steps 1000 --lr 0.05

# Train on webcam (optimize for current frame)
python scripts/train_patch.py --webcam --steps 500
```

> **Note**: This script does NOT vary orientation. For angle-robust patches, use `train_dataset.py --eot` below.

### Train on Dataset with EOT (Recommended for real-world use)


```bash
# Basic training
python scripts/train_dataset.py --dataset data/Birds --steps 2000

# With EOT for angle robustness
python scripts/train_dataset.py --dataset data/Birds --steps 2000 --eot

# Full options
python scripts/train_dataset.py \
    --dataset data/Birds \
    --steps 3000 \
    --batch-size 8 \
    --patch-size 32 \
    --lr 0.01 \
    --eot \
    --n-eot 4 \
    --output my_patches
```

**Parameters:**
| Param | Default | Description |
|-------|---------|-------------|
| `--dataset` | required | Path to image folder |
| `--steps` | 2000 | Training steps |
| `--batch-size` | 4 | Images per batch |
| `--patch-size` | 32 | Patch size in pixels |
| `--lr` | 0.01 | Learning rate (higher = faster, less stable) |
| `--eot` | False | Enable EOT transforms |
| `--n-eot` | 4 | Transforms per image if EOT enabled |
| `--clusters` | 4 | Number of K-means clusters for segmentation |
| `--output` | results | Output directory |
| `--save-every` | 500 | Save checkpoint every N steps |

### Visualize Attack on Image Sequence

After training a patch, visualize its effect on a sequence as a video:

```bash
# Full dashboard (default: mode=all)
python scripts/visualize_sequence.py --dataset data/stuttgart_00 --patch results/universal_patch_final.pt --size 300 --fps 15

# Save as video
python scripts/visualize_sequence.py --dataset data/stuttgart_00 --patch results/universal_patch_final.pt --size 300 --fps 15 --output results/attack_video.mp4
```

**Modes:**

| Mode | Layout | Description |
|------|--------|-------------|
| `all` | 2x4 grid | Full dashboard with all visualizations |
| `pca` | 1x3 | Image + PCA original + PCA attacked |
| `segment` | 1x4 | Image + Seg original + Seg attacked + Diff |
| `trajectory` | 1x2 | Image + PCA token displacement arrows |
| `both` | 1x4 | Image + PCA original + PCA attacked + Distance heatmap |

**Parameters:**
| Param | Default | Description |
|-------|---------|-------------|
| `--dataset` | required | Path to image folder |
| `--patch` | required | Path to trained patch (.pt) |
| `--patch-pos` | 50 50 | Patch position (x, y) |
| `--patch-size` | 32 | Patch size in pixels |
| `--size` | 300 | Visualization panel size |
| `--mode` | all | Visualization mode |
| `--clusters` | 4 | Number of K-means clusters |
| `--fps` | 10 | Playback / output video FPS |
| `--smooth` | off | Use smooth interpolation |
| `--refresh` | 50 | Refresh models every N frames (0 = never) |
| `--output` | None | Save to MP4 instead of display |

**Controls:**
| Key | Action |
|-----|--------|
| `q` | Quit |
| `Space` | Pause / Resume |

### Recommended Datasets

| Dataset | Size | Use Case |
|---------|------|----------|
| `data/Birds/` | Local | Quick testing (already in repo) |
| [ImageNet](https://www.image-net.org/) | 1.2M images | Standard adversarial research |
| [COCO](https://cocodataset.org/) | 330K images | Complex scenes |
| [Pascal VOC](http://host.robots.ox.ac.uk/pascal/VOC/) | 11K images | Smaller, faster to download |
| Custom photos | Any | Your specific use case |

**Dataset structure:**
```
data/
└── YourDataset/
    ├── image1.jpg
    ├── image2.png
    ├── subdir/
    │   ├── image3.jpg
    │   └── ...
    └── ...
```

---

## Orientation & Perspective Variation

### How It Works

Real-world patches are viewed from different angles. EOT simulates this during training:

```
Original Patch          →    Transformed Patch
┌──────────┐                 ╱────────────╲
│          │    rotate      ╱              ╲
│  PATCH   │    + tilt     │    PATCH      │
│          │    + scale     ╲              ╱
└──────────┘                 ╲────────────╱
```

### Transformation Types

1. **Rotation** (yaw): Patch turns left/right (-30° to +30°)
2. **Perspective** (pitch): Viewed from above/below
3. **Scale**: Distance variation (0.8x to 1.2x)
4. **Color jitter**: Lighting changes (brightness, contrast)

### API Usage

```python
from src.attacks import EOTAttack, EOTConfig

# Configure transforms
config = EOTConfig(
    num_samples=8,           # Transforms per optimization step
    rotation_range=30,       # Max rotation degrees
    scale_range=(0.8, 1.2),  # Scale variation
    perspective_strength=0.2,# Perspective distortion
    brightness_range=0.2,    # Lighting variation
    contrast_range=0.2,
)

# Train EOT-robust patch
eot_attack = EOTAttack(model, config)
patch = eot_attack.train(
    images,           # List of training images
    patch_size=32,
    steps=1000,
    lr=0.01,
)
```

### Simulate Oriented Patch in Video

Apply a trained patch with 3D perspective to live video:

```python
from src.attacks import apply_patch_with_perspective

# Apply patch with specific orientation
image_with_patch = apply_patch_with_perspective(
    image,
    patch,
    position=(100, 100),
    yaw=15,        # Degrees rotation
    pitch=10,      # Degrees tilt
    roll=0,        # Degrees roll
    scale=1.0,
)
```

---

## Video Processing

### Extract Features from Video

```bash
python src/video_processor.py path/to/video.mp4 --batch-size 8
```

### Attack a Video

```bash
python src/attacks/video_attack.py path/to/video.mp4 \
    --patch-size 32 \
    --steps 500 \
    --output attacked_video.mp4
```

### Python API

```python
from src.video_processor import VideoProcessor, VideoConfig
from src.attacks import VideoAttacker, AttackConfig

# Feature extraction
processor = VideoProcessor()
results = processor.process_video("video.mp4")
print(f"CLS tokens: {results['cls_tokens'].shape}")

# Keyframe detection
keyframes = processor.find_keyframes(results['cls_tokens'])

# Video attack
config = AttackConfig(patch_size=32, steps=500)
attacker = VideoAttacker(attack_config=config)
attack_results = attacker.attack_video("video.mp4", output_path="attacked.mp4")
```

---

## Static Image Attacks

### Embedding Space Attack

Maximize L2 distance between original and attacked embeddings:

```bash
python examples/demo_dinov3.py
```

### Segmentation Coherence Attack

Destroy spatial coherence in unsupervised segmentation:

```bash
python examples/demo_segmentation_attack.py
```

---

## How It Works

### DINOv3 Token Extraction

```
Image (224x224)
     ↓
Split into 14x14 patches (16x16 pixels each)
     ↓
DINOv3 Vision Transformer
     ↓
196 patch tokens (each 384-dimensional)
     ↓
Tokens encode semantic information
```

### Adversarial Patch Attack

```
1. Get reference tokens (without patch)
2. Initialize random patch
3. For each optimization step:
   - Apply patch to image
   - Extract tokens
   - Loss = -MSE(tokens_ref, tokens_adv)  # Maximize distance
   - Backprop & update patch
4. Clamp patch to [0, 1]
```

### Segmentation

Segmentation is performed by clustering the patch tokens:

- **PCA**: Project 384D tokens to 3D RGB
- **K-Means**: Cluster tokens, assign colors

Tokens from the same object are close in embedding space → same cluster → same color.

---

## Attack Types

### Embedding Attacks

```python
# Maximize distance
attack = DINOv3PatchAttack(model, patch_size=50)
results = attack.attack_embedding(image, attack_type='maximize_distance')

# Maximize/Minimize norm
results = attack.attack_embedding(image, attack_type='maximize_norm')
results = attack.attack_embedding(image, attack_type='minimize_norm')
```

### Segmentation Attacks

```python
# Entropy attack (fragment segments)
attack = SegmentationCoherenceAttack(model, segmenter, patch_size=50)
results = attack.attack(image, attack_mode='entropy')

# Coherence attack (reduce similarity)
results = attack.attack(image, attack_mode='coherence')

# Boundary attack (maximize boundary length)
results = attack.attack(image, attack_mode='boundary')
```

---

## Metrics

### Embedding Attacks
| Metric | Description |
|--------|-------------|
| L2 distance | Euclidean distance between token sets |
| Cosine similarity | Angular similarity |
| MSE | Mean squared error between tokens |

### Segmentation Attacks
| Metric | Description |
|--------|-------------|
| Coherence degradation | Reduction in intra-segment similarity |
| Boundary increase | Additional boundary pixels created |
| IoU | Intersection over Union with original segmentation |

---

## Performance (Mac M3 Max)

| Task | FPS |
|------|-----|
| Webcam segmentation | ~15-25 |
| Interactive attack | ~10-15 |
| Video processing (batch=8) | ~50-80 |
| Patch optimization | ~3-5 it/s |

---

## Saved Patches

Optimized patches are saved to `results/`:

```bash
results/
├── universal_patch.pt      # Latest universal patch
├── patch_1234567890.pt     # Timestamped patches (press 'w')
└── *.png                   # Visualizations
```

Load a saved patch:
```python
patch = torch.load("results/universal_patch.pt")
```

---

## Development

```bash
# Run tests
pytest

# Format code
black src/ examples/
ruff check src/ examples/
```

---

## Research Ideas

See [docs/segmentation_ideas.md](docs/segmentation_ideas.md) for detailed research directions:

1. Unsupervised segmentation + spatial coherence attack
2. Patch attack to hijack segmentation boundaries
3. Combined SAM + DINOv3 attack
4. Targeted mask theft
5. Object hallucination
6. Temporal consistency attacks (video)
