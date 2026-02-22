import torch

CITYSCAPES_IMAGES = "data/leftImg8bit_trainvaltest/leftImg8bit/train"
CITYSCAPES_LABELS = "data/gtFine_trainvaltest/gtFine/train"
DATASET = "data/leftImg8bit_trainvaltest/leftImg8bit/train"
VIZ_DATASET = "data/stuttgart/stuttgart_02"
CLASSIFIER = "results/classifier.pt"
PATCH = "results/targeted_patch_best.pt"
OUTPUT_DIR = "results"

IMG_SIZE = 896

CLF_EPOCHS = 200
CLF_LR = 0.001

SOURCE_CLASS = 11
TARGET_CLASS = -1
ATTACK_STEPS = 4000
ATTACK_LR = 0.05
ATTACK_BATCH_SIZE = 4
ATTACK_MIN_SOURCE_TOKENS = 10
PATCH_SIZE = 140
PATCH_RES = 512
PATCH_PERSPECTIVE_MIN_SCALE = 0.3
PATCH_MIN_ROW_RATIO = 0.3
PATCH_Y_RATIO = 0.85
PATCH_SAVE_EVERY = 200

VIZ_SIZE = 720
VIZ_EVERY = 10
VIZ_SEQ_SIZE = 450
FPS = 10
REFRESH = 50
FOCUS_CLASSES = [0, 11, 13]


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
