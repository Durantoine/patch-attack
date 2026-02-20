"""Default configuration for all scripts.

Modify these values to change defaults without CLI arguments.
"""

# === Paths ===
CITYSCAPES_IMAGES: str = "data/leftImg8bit_trainvaltest/leftImg8bit/train"
CITYSCAPES_LABELS: str = "data/gtFine_trainvaltest-2/gtFine/train"
DATASET: str = "data/leftImg8bit_trainvaltest/leftImg8bit/train"   # training dataset (Cityscapes)
VIZ_DATASET: str = "data/stuttgart_02"                             # sequence for visualization
CLASSIFIER: str = "results/classifier.pt"
OUTPUT_DIR: str = "results"

# === Model ===
IMG_SIZE: int = 672          # 224=14x14, 448=28x28, 672=42x42

# === Classifier training (train_classifier.py) ===
CLF_EPOCHS: int = 20
CLF_LR: float = 0.001

# === Attack (attack_classifier.py) ===
SOURCE_CLASS: int = 11       # person
TARGET_CLASS: int = -1       # -1 = untargeted (any misclassification)
ATTACK_STEPS: int = 3000
ATTACK_LR: float = 0.05
ATTACK_BATCH_SIZE: int = 4
ATTACK_MIN_SOURCE_TOKENS: int = 10  # Min source-class tokens per image to include in training
PATCH_SIZE: int = 132        # Size on ismage (pixels) — ~17% of IMG_SIZE
PATCH_RES: int = 256         # Internal patch resolution (optimized pixels)
PATCH_PERSPECTIVE_MIN_SCALE: float = 0.3  # Min scale at top (far) vs bottom (near)
PATCH_MIN_ROW_RATIO: float = 0.3        # Min vertical position — near horizon, below sky (~35% from top)
PATCH_Y_RATIO: float = 0.85             # Horizontal position as fraction of image width — 0.28=left sidewalk, 0.72=right sidewalk
PATCH_SAVE_EVERY: int = 200             # Save patch snapshot every N steps for evolution replay (0=off)

# === Visualization ===
VIZ_SIZE: int = 500          # Panel size for training viz
VIZ_EVERY: int = 10          # Show live viz every N steps (0=off)
VIZ_SEQ_SIZE: int = 300      # Panel size for sequence visualization
FPS: int = 10                # Playback FPS
CLUSTERS: int = 4            # K-means clusters
REFRESH: int = 50            # Refresh models every N frames
FOCUS_CLASSES: list[int] = [0, 11, 13]  # road, person, car
PATCH_POS: list[int] = [50, 50]         # Default patch position
