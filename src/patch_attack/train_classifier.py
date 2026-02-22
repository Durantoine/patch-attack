from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from .models.dinov3_loader import load_dinov3
from .utils.config import (
    CITYSCAPES_IMAGES,
    CITYSCAPES_LABELS,
    CLASSIFIER,
    CLF_EPOCHS,
    CLF_LR,
    IMG_SIZE,
    VIZ_SIZE,
    get_device,
)
from .utils.viz import CLASS_NAMES, colorize_preds, create_legend

LABEL_MAP = {
    7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5, 19: 6, 20: 7, 21: 8, 22: 9,
    23: 10, 24: 11, 25: 12, 26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18,
}
IGNORE = 255


def labels_to_tokens(label_img: Image.Image, grid: int) -> np.ndarray:
    arr = np.array(label_img)
    mapped = np.full_like(arr, IGNORE)
    for lid, tid in LABEL_MAP.items():
        mapped[arr == lid] = tid
    ph, pw = arr.shape[0] // grid, arr.shape[1] // grid
    tokens = np.full(grid * grid, IGNORE, dtype=np.int64)
    for i in range(grid):
        for j in range(grid):
            valid = mapped[i * ph : (i + 1) * ph, j * pw : (j + 1) * pw]
            valid = valid[valid != IGNORE]
            if len(valid):
                tokens[i * grid + j] = np.bincount(valid).argmax()
    return tokens


def find_pairs(images_dir: str, labels_dir: str) -> list:
    pairs = []
    for city in sorted(Path(labels_dir).iterdir()):
        if not city.is_dir():
            continue
        for lbl in sorted(city.glob("*_labelIds.png")):
            img = (
                Path(images_dir)
                / city.name
                / lbl.name.replace("_gtFine_labelIds.png", "_leftImg8bit.png")
            )
            if img.exists():
                pairs.append((img, lbl))
    return pairs


class ClassifierTrainer:
    def __init__(self) -> None:
        self.device = get_device()
        self.grid = IMG_SIZE // 16
        print(f"Image {IMG_SIZE}×{IMG_SIZE} → {self.grid}×{self.grid} tokens | {self.device}")
        self.model = load_dinov3(self.device)
        self.clf = nn.Linear(384, 19).to(self.device)
        self.opt = torch.optim.Adam(self.clf.parameters(), lr=CLF_LR)
        self.img_tf = transforms.Compose([
            transforms.Resize(IMG_SIZE + 32),
            transforms.CenterCrop(IMG_SIZE),
            transforms.ToTensor(),
        ])
        self.lbl_tf = transforms.Compose([
            transforms.Resize(IMG_SIZE, interpolation=transforms.InterpolationMode.NEAREST),
            transforms.CenterCrop(IMG_SIZE),
        ])

    def load_data(self) -> tuple:
        pairs = find_pairs(CITYSCAPES_IMAGES, CITYSCAPES_LABELS)
        print(f"{len(pairs)} paires image-label")
        if not pairs:
            return None, None, []

        step_viz = max(1, len(pairs) // 4)
        viz_idx = {i * step_viz for i in range(min(4, len(pairs)))}
        viz_samples, all_tokens, all_labels = [], [], []

        for i, (img_path, lbl_path) in enumerate(tqdm(pairs, desc="Extraction tokens")):
            img = self.img_tf(Image.open(img_path).convert("RGB")).unsqueeze(0).to(self.device)
            lbl = labels_to_tokens(self.lbl_tf(Image.open(lbl_path)), self.grid)
            with torch.no_grad():
                tokens = F.normalize(
                    self.model.get_intermediate_layers(img, n=1)[0][0, 1:], dim=-1
                )
            all_tokens.append(tokens.cpu())
            all_labels.append(torch.from_numpy(lbl[: tokens.shape[0]]))
            if i in viz_idx:
                viz_samples.append((img[0].cpu(), tokens.cpu(), lbl[: tokens.shape[0]]))

        tokens_cat = torch.cat(all_tokens)
        labels_cat = torch.cat(all_labels)
        valid = labels_cat != IGNORE
        tokens_cat = tokens_cat[valid].to(self.device)
        labels_cat = labels_cat[valid].to(self.device).long()
        print(f"{len(tokens_cat)} tokens valides")
        return tokens_cat, labels_cat, viz_samples

    def train_epoch(
        self, epoch: int, tokens_cat: torch.Tensor, labels_cat: torch.Tensor
    ) -> tuple[float, float]:
        perm = torch.randperm(len(tokens_cat))
        correct, total_loss = 0, 0.0
        for i in range(0, len(tokens_cat), 4096):
            idx = perm[i : i + 4096]
            logits = self.clf(tokens_cat[idx])
            loss = F.cross_entropy(logits, labels_cat[idx])
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            total_loss += loss.item() * len(idx)
            correct += (logits.argmax(1) == labels_cat[idx]).sum().item()
        avg_loss = total_loss / len(tokens_cat)
        acc = correct / len(tokens_cat)
        print(f"Époque {epoch + 1}/{CLF_EPOCHS} | Loss: {avg_loss:.4f} | Acc: {acc:.3f}")
        return avg_loss, acc

    def render_frame(
        self, epoch: int, avg_loss: float, acc: float, viz_samples: list
    ) -> bool:
        sz = VIZ_SIZE
        rows = []
        for img_t, tok, lbl in viz_samples:
            img_np = (np.clip(img_t.permute(1, 2, 0).numpy(), 0, 1) * 255).astype(np.uint8)
            img_d = cv2.cvtColor(cv2.resize(img_np, (sz, sz)), cv2.COLOR_RGB2BGR)
            gt_vis = colorize_preds(lbl, self.grid, sz)
            with torch.no_grad():
                pred_vis = colorize_preds(
                    self.clf(tok.to(self.device)).argmax(-1).cpu().numpy(), self.grid, sz
                )
            rows.append(np.hstack([img_d, gt_vis, pred_vis]))
        grid_vis = np.vstack(rows)
        frame = np.hstack([grid_vis, create_legend(grid_vis.shape[0])])
        for i, label in enumerate(["Image", "GT", "Prédiction"]):
            cv2.putText(
                frame, label, (i * sz + 5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1
            )
        cv2.putText(
            frame,
            f"Époque {epoch + 1}/{CLF_EPOCHS} | Loss: {avg_loss:.4f} | Acc: {acc:.1%}",
            (10, grid_vis.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            1,
        )
        cv2.imshow("Classifier Training", frame)
        return cv2.waitKey(500) & 0xFF == ord("q")

    def save(self) -> None:
        Path(CLASSIFIER).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"state_dict": self.clf.state_dict(), "class_names": CLASS_NAMES, "img_size": IMG_SIZE},
            CLASSIFIER,
        )
        print(f"Sauvegardé → {CLASSIFIER}")

    def run(self) -> None:
        tokens_cat, labels_cat, viz_samples = self.load_data()
        if tokens_cat is None:
            return
        cv2.namedWindow("Classifier Training", cv2.WINDOW_NORMAL)
        for epoch in range(CLF_EPOCHS):
            avg_loss, acc = self.train_epoch(epoch, tokens_cat, labels_cat)
            if self.render_frame(epoch, avg_loss, acc, viz_samples):
                print("Interrompu.")
                break
        cv2.destroyAllWindows()
        self.save()


def main() -> None:
    ClassifierTrainer().run()


if __name__ == "__main__":
    main()
