from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

CLASS_NAMES: list[str] = [
    "road",
    "sidewalk",
    "building",
    "wall",
    "fence",
    "pole",
    "traffic_light",
    "traffic_sign",
    "vegetation",
    "terrain",
    "sky",
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
]

CLASS_NAMES_SHORT: list[str] = [
    "road",
    "sidew",
    "build",
    "wall",
    "fence",
    "pole",
    "TL",
    "sign",
    "veg",
    "terr",
    "sky",
    "pers",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "moto",
    "bike",
]

CITYSCAPES_COLORS: np.ndarray = np.array(
    [
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [70, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32],
    ],
    dtype=np.uint8,
)

OTHER_COLOR: np.ndarray = np.array([80, 80, 80], dtype=np.uint8)

DEFAULT_FOCUS_CLASSES: list[int] = [0, 11, 13]  # road, person, car


def colorize_preds(
    preds: np.ndarray, grid: int, size: int, focus_classes: list[int] | None = None
) -> np.ndarray:
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


def create_legend(
    size: int, present_classes: np.ndarray | None = None, focus_classes: list[int] | None = None
) -> np.ndarray:
    legend_w = size // 2
    legend = np.zeros((size, legend_w, 3), dtype=np.uint8)
    legend[:] = 30
    if focus_classes is not None:
        classes: list[int] = sorted(focus_classes) + [-1]
    elif present_classes is not None:
        classes = sorted(set(int(c) for c in present_classes))
    else:
        classes = sorted(DEFAULT_FOCUS_CLASSES) + [-1]
    box = 24
    font_scale = 0.7
    spacing = min(size // (len(classes) + 1), box + 20)
    y_start = max(15, (size - len(classes) * spacing) // 2)
    for idx, cls_id in enumerate(classes):
        y = y_start + idx * spacing
        if y + box > size:
            break
        if cls_id == -1:
            rgb: np.ndarray = OTHER_COLOR
            name = "other"
        else:
            rgb = CITYSCAPES_COLORS[cls_id]
            name = CLASS_NAMES_SHORT[cls_id]
        color: tuple[int, int, int] = (int(rgb[2]), int(rgb[1]), int(rgb[0]))
        cv2.rectangle(legend, (10, y), (10 + box, y + box), color, -1)
        cv2.rectangle(legend, (10, y), (10 + box, y + box), (200, 200, 200), 1)
        cv2.putText(
            legend,
            name,
            (10 + box + 8, y + box - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (220, 220, 220),
            1,
        )
    return legend


def compute_perspective_size(x: int, img_size: int, max_size: int, min_scale: float) -> int:
    t = min(1.0, (x + max_size / 2) / img_size)
    scale = min_scale + (1.0 - min_scale) * t
    return max(1, int(max_size * scale))


def patch_to_img(patch: torch.Tensor, size: int) -> np.ndarray:
    img = np.clip(patch.detach().cpu().permute(1, 2, 0).numpy(), 0, 1)
    img = (img * 255).astype(np.uint8)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    return np.asarray(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def tensor_to_bgr(t: torch.Tensor) -> np.ndarray:
    arr = t.detach().cpu().permute(1, 2, 0).numpy().clip(0, 1)
    return cv2.cvtColor((arr * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)


def letterbox(img_bgr: np.ndarray, size: int) -> tuple[np.ndarray, float, int, int]:
    h, w = img_bgr.shape[:2]
    scale = min(size / w, size / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    pad_x = (size - new_w) // 2
    pad_y = (size - new_h) // 2
    canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized
    return canvas, scale, pad_x, pad_y


def apply_patch(image: torch.Tensor, patch: torch.Tensor, pos: tuple[int, int]) -> torch.Tensor:
    x, y = pos
    out = image.clone()
    xe = min(x + patch.shape[1], image.shape[1])
    ye = min(y + patch.shape[2], image.shape[2])
    out[:, x:xe, y:ye] = patch[:, : xe - x, : ye - y]
    return out


def resize_patch(patch: torch.Tensor, size: int) -> torch.Tensor:
    return F.interpolate(
        patch.unsqueeze(0), (size, size), mode="bilinear", align_corners=False
    ).squeeze(0)


def make_evolution_video(
    evo_dir: Path,
    evo_steps: list,
    history: list,
    total_steps: int,
    out_dir: Path,
    src: str,
    tgt: str,
) -> None:
    sz = 256
    video_path = out_dir / "patch_evolution.mp4"
    fps_evo = max(2, min(15, len(evo_steps) // 5 + 1))
    fourcc: int = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
    _w = cv2.VideoWriter(str(video_path), fourcc, fps_evo, (sz + 400, sz))
    writer: cv2.VideoWriter | None = _w if _w.isOpened() else None
    for s in evo_steps:
        pt = evo_dir / f"patch_step_{s:05d}.pt"
        if not pt.exists():
            continue
        patch_img = patch_to_img(torch.load(pt, map_location="cpu", weights_only=True), sz)
        fr = history[min(s - 1, len(history) - 1)] if history else 0.0
        panel = np.full((sz, 400, 3), 25, dtype=np.uint8)
        cv2.putText(
            panel,
            f"Step {s}/{total_steps}",
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (220, 220, 220),
            2,
        )
        cv2.putText(
            panel,
            f"Attack: {src} -> {tgt}",
            (10, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (180, 180, 180),
            1,
        )
        cv2.rectangle(panel, (10, 110), (370, 150), (60, 60, 60), -1)
        if fr > 0:
            cv2.rectangle(panel, (10, 110), (10 + int(360 * fr), 150), (0, 200, 80), -1)
        cv2.putText(
            panel,
            f"Fooling: {fr:.0%}",
            (10, 180),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 220, 100),
            2,
        )
        frame = np.hstack([patch_img, panel])
        if writer:
            writer.write(frame)
    if writer:
        writer.release()
        print(f"Vidéo évolution : {video_path}")
    n = min(20, len(evo_steps))
    idx = [int(i * (len(evo_steps) - 1) / max(1, n - 1)) for i in range(n)]
    cols, rows = min(5, n), (n + 4) // 5
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.5))
    axes_flat = np.array(axes).flatten() if n > 1 else [axes]
    for ai, si in enumerate(idx):
        s = evo_steps[si]
        pt = Path(evo_dir) / f"patch_step_{s:05d}.pt"
        ax = axes_flat[ai]
        if pt.exists():
            p = torch.load(pt, map_location="cpu", weights_only=True)
            fr = history[min(s - 1, len(history) - 1)] if history else 0.0
            ax.imshow(np.clip(p.permute(1, 2, 0).numpy(), 0, 1))
            ax.set_title(f"Step {s}\nFR {fr:.0%}", fontsize=8)
        ax.axis("off")
    for ax in axes_flat[n:]:
        ax.axis("off")
    fig.suptitle(f"Évolution — {src} → {tgt}", fontsize=12)
    plt.tight_layout()
    grid_path = out_dir / "patch_evolution_grid.png"
    plt.savefig(grid_path, dpi=150)
    plt.close(fig)
    print(f"Grille évolution  : {grid_path}")
