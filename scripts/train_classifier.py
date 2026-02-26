"""Train linear probe on DINOv3 tokens with Cityscapes labels.

Usage:
    python scripts/train_classifier.py --images data/leftImg8bit/train --labels data/gtFine_trainvaltest-2/gtFine/train
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
import argparse
from tqdm import tqdm
from models.dinov3_loader import load_dinov3
from utils.viz import CLASS_NAMES, colorize_preds, create_legend
from utils.config import (
    CITYSCAPES_IMAGES, CITYSCAPES_LABELS, IMG_SIZE, CLF_EPOCHS, CLF_LR,
    CLASSIFIER, VIZ_SIZE,
)

LABEL_MAP: dict[int, int] = {
    7:0, 8:1, 11:2, 12:3, 13:4, 17:5, 19:6, 20:7,
    21:8, 22:9, 23:10, 24:11, 25:12, 26:13, 27:14,
    28:15, 31:16, 32:17, 33:18
}
IGNORE: int = 255

def get_device() -> torch.device:
    if torch.cuda.is_available(): return torch.device('cuda')
    if torch.backends.mps.is_available(): return torch.device('mps')
    return torch.device('cpu')

def load_model(device: torch.device) -> nn.Module:
    w: Path = Path(__file__).parent.parent / "src/models/weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
    m: nn.Module = load_dinov3(checkpoint_path=str(w), device=str(device))
    m.eval()
    for p in m.parameters(): p.requires_grad = False
    return m

def labels_to_tokens(label_img: Image.Image, grid: int = 14) -> np.ndarray:
    """Pixel labels -> token labels via majority vote per 16x16 patch."""
    arr: np.ndarray = np.array(label_img)
    mapped: np.ndarray = np.full_like(arr, IGNORE)
    for lid, tid in LABEL_MAP.items():
        mapped[arr == lid] = tid
    ph: int = arr.shape[0] // grid
    pw: int = arr.shape[1] // grid
    tokens: np.ndarray = np.full(grid * grid, IGNORE, dtype=np.int64)
    for i in range(grid):
        for j in range(grid):
            patch: np.ndarray = mapped[i*ph:(i+1)*ph, j*pw:(j+1)*pw]
            valid: np.ndarray = patch[patch != IGNORE]
            if len(valid) > 0:
                tokens[i * grid + j] = np.bincount(valid).argmax()
    return tokens

def find_pairs(images_dir: str, labels_dir: str) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    for city in sorted(Path(labels_dir).iterdir()):
        if not city.is_dir(): continue
        for lbl in sorted(city.glob("*_labelIds.png")):
            img: Path = Path(images_dir) / city.name / lbl.name.replace("_gtFine_labelIds.png", "_leftImg8bit.png")
            if img.exists():
                pairs.append((img, lbl))
    return pairs

def main() -> None:
    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    parser.add_argument('--images', default=CITYSCAPES_IMAGES)
    parser.add_argument('--labels', default=CITYSCAPES_LABELS)
    parser.add_argument('--epochs', type=int, default=CLF_EPOCHS)
    parser.add_argument('--lr', type=float, default=CLF_LR)
    parser.add_argument('--img-size', type=int, default=IMG_SIZE)
    parser.add_argument('--output', default=CLASSIFIER)
    parser.add_argument('--viz-size', type=int, default=VIZ_SIZE)
    args: argparse.Namespace = parser.parse_args()

    device: torch.device = get_device()
    img_size: int = args.img_size
    grid: int = img_size // 16
    print(f"Image size: {img_size}x{img_size} -> {grid}x{grid} tokens")

    pairs: list[tuple[Path, Path]] = find_pairs(args.images, args.labels)
    print(f"{len(pairs)} image-label pairs | Device: {device}")
    if not pairs: return

    model: nn.Module = load_model(device)
    img_tf: transforms.Compose = transforms.Compose([transforms.Resize(img_size + 32), transforms.CenterCrop(img_size), transforms.ToTensor()])
    lbl_tf: transforms.Compose = transforms.Compose([
        transforms.Resize(img_size, interpolation=transforms.InterpolationMode.NEAREST),
        transforms.CenterCrop(img_size)
    ])

    # Pick 4 evenly spaced samples for visualization
    n_viz: int = 4
    viz_indices: set[int] = set()
    if len(pairs) >= n_viz:
        step_viz: int = len(pairs) // n_viz
        viz_indices = {i * step_viz for i in range(n_viz)}
    else:
        viz_indices = set(range(len(pairs)))
    viz_samples: list[tuple[torch.Tensor, torch.Tensor, np.ndarray]] = []

    all_tokens: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    for pair_idx, (img_path, lbl_path) in enumerate(tqdm(pairs, desc="Extracting tokens")):
        img: torch.Tensor = img_tf(Image.open(img_path).convert('RGB')).unsqueeze(0).to(device)
        lbl: np.ndarray = labels_to_tokens(lbl_tf(Image.open(lbl_path)), grid=grid)
        with torch.no_grad():
            tokens: torch.Tensor = model.get_intermediate_layers(img, n=1)[0][0]
        n_tokens: int = tokens.shape[0]
        all_tokens.append(tokens.cpu())
        all_labels.append(torch.from_numpy(lbl[:n_tokens]))
        if pair_idx in viz_indices:
            viz_samples.append((img[0].cpu(), tokens.cpu(), lbl[:n_tokens]))

    tokens_cat: torch.Tensor = torch.cat(all_tokens)
    labels_cat: torch.Tensor = torch.cat(all_labels)
    valid: torch.Tensor = labels_cat != IGNORE
    tokens_cat, labels_cat = tokens_cat[valid], labels_cat[valid].long()
    print(f"{len(tokens_cat)} valid tokens")

    clf: nn.Linear = nn.Linear(384, 19).to(device)
    opt: torch.optim.Adam = torch.optim.Adam(clf.parameters(), lr=args.lr)

    viz_on: bool = args.viz_size > 0
    if viz_on:
        cv2.namedWindow("Classifier Training", cv2.WINDOW_NORMAL)
        cv2.moveWindow("Classifier Training", 100, 100)

    for epoch in range(args.epochs):
        perm: torch.Tensor = torch.randperm(len(tokens_cat))
        correct: int = 0
        total_loss: float = 0.0
        for i in range(0, len(tokens_cat), 4096):
            idx: torch.Tensor = perm[i:i+4096]
            batch_tok: torch.Tensor = tokens_cat[idx].to(device)
            batch_lab: torch.Tensor = labels_cat[idx].to(device)
            logits: torch.Tensor = clf(batch_tok)
            loss: torch.Tensor = F.cross_entropy(logits, batch_lab)
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item() * len(idx)
            correct += (logits.argmax(1) == batch_lab).sum().item()
        acc: float = correct / len(tokens_cat)
        avg_loss: float = total_loss / len(tokens_cat)
        print(f"Epoch {epoch+1}/{args.epochs} | Loss: {avg_loss:.4f} | Acc: {acc:.3f}")

        # Live visualization: each sample = one row [Image | GT | Predicted]
        if viz_on:
            sz: int = args.viz_size
            rows: list[np.ndarray] = []
            for img_tensor, sample_tokens, sample_labels in viz_samples:
                img_np: np.ndarray = img_tensor.permute(1, 2, 0).numpy()
                img_np = (np.clip(img_np, 0, 1) * 255).astype(np.uint8)
                img_display: np.ndarray = cv2.resize(img_np, (sz, sz))
                img_display = cv2.cvtColor(img_display, cv2.COLOR_RGB2BGR)
                gt_vis: np.ndarray = colorize_preds(sample_labels, grid, sz)
                with torch.no_grad():
                    pred_p: np.ndarray = clf(sample_tokens.to(device)).argmax(-1).cpu().numpy()
                pred_vis: np.ndarray = colorize_preds(pred_p, grid, sz)
                rows.append(np.hstack([img_display, gt_vis, pred_vis]))

            grid_vis: np.ndarray = np.vstack(rows)
            legend_vis: np.ndarray = create_legend(grid_vis.shape[0])
            frame: np.ndarray = np.hstack([grid_vis, legend_vis])
            # Column headers
            for i, label in enumerate(["Image", "GT", "Predicted"]):
                cv2.putText(frame, label, (i * sz + 5, 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            cv2.putText(frame, f"Epoch {epoch+1}/{args.epochs} | Loss: {avg_loss:.4f} | Acc: {acc:.1%}",
                       (10, grid_vis.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

            cv2.imshow("Classifier Training", frame)
            key: int = cv2.waitKey(500) & 0xFF
            if key == ord('q'):
                print("Training interrupted by user.")
                break

    if viz_on:
        cv2.destroyAllWindows()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save({'state_dict': clf.state_dict(), 'class_names': CLASS_NAMES, 'img_size': img_size}, args.output)
    print(f"Saved to {args.output}")

if __name__ == "__main__":
    main()
