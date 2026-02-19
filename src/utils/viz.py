"""Shared visualization utilities for Cityscapes segmentation display."""

import numpy as np
import cv2
import torch

CLASS_NAMES: list[str] = [
    "road","sidewalk","building","wall","fence","pole",
    "traffic_light","traffic_sign","vegetation","terrain","sky",
    "person","rider","car","truck","bus","train","motorcycle","bicycle"
]

CLASS_NAMES_SHORT: list[str] = [
    "road","sidew","build","wall","fence","pole",
    "TL","sign","veg","terr","sky","pers","rider","car",
    "truck","bus","train","moto","bike"
]

CITYSCAPES_COLORS: np.ndarray = np.array([
    [128,64,128],[244,35,232],[70,70,70],[102,102,156],[190,153,153],
    [153,153,153],[250,170,30],[220,220,0],[107,142,35],[152,251,152],
    [70,130,180],[220,20,60],[255,0,0],[0,0,142],[0,0,70],
    [0,60,100],[0,80,100],[0,0,230],[119,11,32],
], dtype=np.uint8)

OTHER_COLOR: np.ndarray = np.array([80, 80, 80], dtype=np.uint8)

DEFAULT_FOCUS_CLASSES: list[int] = [0, 11, 13]  # road, person, car


def colorize_preds(preds: np.ndarray, grid: int, size: int,
                   focus_classes: list[int] | None = None) -> np.ndarray:
    """Prediction array -> colored BGR image.

    If focus_classes is set, only those classes are colored; the rest is gray.
    """
    if focus_classes is None:
        focus_classes = DEFAULT_FOCUS_CLASSES
    n = grid * grid
    if len(preds) < n:
        preds = np.concatenate([preds, np.full(n - len(preds), preds[-1])])
    seg = preds[:n].reshape(grid, grid)
    colored = np.full((grid, grid, 3), 80, dtype=np.uint8)
    for i in focus_classes:
        colored[seg == i] = CITYSCAPES_COLORS[i]
    vis = cv2.resize(colored, (size, size), interpolation=cv2.INTER_NEAREST)
    return cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)


def create_legend(size: int, present_classes: np.ndarray | None = None,
                  focus_classes: list[int] | None = None) -> np.ndarray:
    """Create a vertical color legend.

    If focus_classes is set, show those + 'other'.
    Else if present_classes is set, show only those.
    Else use DEFAULT_FOCUS_CLASSES.
    """
    legend_w: int = size // 2
    legend: np.ndarray = np.zeros((size, legend_w, 3), dtype=np.uint8)
    legend[:] = 30
    if focus_classes is not None:
        classes: list[int] = sorted(focus_classes) + [-1]
    elif present_classes is not None:
        classes = sorted(set(int(c) for c in present_classes))
    else:
        classes = sorted(DEFAULT_FOCUS_CLASSES) + [-1]
    n: int = len(classes)
    box: int = 24
    font_scale: float = 0.7
    spacing: int = min(size // (n + 1), box + 20)
    y_start: int = max(15, (size - n * spacing) // 2)
    for idx, cls_id in enumerate(classes):
        y: int = y_start + idx * spacing
        if y + box > size:
            break
        if cls_id == -1:
            rgb: np.ndarray = OTHER_COLOR
            name: str = "other"
        else:
            rgb = CITYSCAPES_COLORS[cls_id]
            name = CLASS_NAMES_SHORT[cls_id]
        color: tuple[int, ...] = (int(rgb[2]), int(rgb[1]), int(rgb[0]))  # RGB -> BGR
        cv2.rectangle(legend, (10, y), (10 + box, y + box), color, -1)
        cv2.rectangle(legend, (10, y), (10 + box, y + box), (200, 200, 200), 1)
        cv2.putText(legend, name, (10 + box + 8, y + box - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (220, 220, 220), 1)
    return legend


def patch_to_img(patch: torch.Tensor, size: int) -> np.ndarray:
    """Patch tensor (3,H,W) -> BGR image."""
    img = np.clip(patch.detach().cpu().permute(1, 2, 0).numpy(), 0, 1)
    img = (img * 255).astype(np.uint8)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
