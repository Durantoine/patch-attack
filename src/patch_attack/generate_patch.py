from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from .models.dinov3_loader import load_dinov3
from .utils.config import (
    ATTACK_BATCH_SIZE,
    ATTACK_LR,
    ATTACK_MIN_SOURCE_TOKENS,
    ATTACK_STEPS,
    CLASSIFIER,
    DATASET,
    OUTPUT_DIR,
    PATCH_MIN_ROW_RATIO,
    PATCH_PERSPECTIVE_MIN_SCALE,
    PATCH_RES,
    PATCH_SAVE_EVERY,
    PATCH_SIZE,
    PATCH_Y_RATIO,
    SOURCE_CLASS,
    TARGET_CLASS,
    VIZ_EVERY,
    VIZ_SIZE,
    get_device,
)
from .utils.viz import (
    CLASS_NAMES,
    colorize_preds,
    compute_perspective_size,
    create_legend,
    make_evolution_video,
    patch_to_img,
)


def apply_patch(
    images: torch.Tensor, patch: torch.Tensor, sizes: list, positions: list
) -> torch.Tensor:
    out = images.clone()
    for i, (ps, (x, y)) in enumerate(zip(sizes, positions)):
        p = F.interpolate(patch.unsqueeze(0), (ps, ps), mode="bilinear", align_corners=False)[0]
        xe, ye = min(x + ps, images.shape[2]), min(y + ps, images.shape[3])
        out[i, :, x:xe, y:ye] = p[:, : xe - x, : ye - y]
    return out


class PatchAttack:
    def __init__(self) -> None:
        self.device = get_device()
        self.untargeted = TARGET_CLASS == -1
        self.src_name = CLASS_NAMES[SOURCE_CLASS]
        self.tgt_name = "any" if self.untargeted else CLASS_NAMES[TARGET_CLASS]
        print(
            f"Attaque : {self.src_name} → {self.tgt_name} | {self.device} | "
            f"patch {PATCH_RES}→{PATCH_SIZE}px | perspective {PATCH_PERSPECTIVE_MIN_SCALE:.0%}–100%"
        )
        self.model = load_dinov3(self.device)
        clf_data = torch.load(CLASSIFIER, map_location=self.device, weights_only=False)
        self.clf = nn.Linear(384, 19).to(self.device)
        self.clf.load_state_dict(clf_data["state_dict"])
        self.clf.eval()
        self.img_size: int = clf_data.get("img_size", 224)
        self.grid = self.img_size // 16
        print(f"Image {self.img_size}×{self.img_size} → {self.grid}×{self.grid} tokens")
        self.tf = transforms.Compose(
            [
                transforms.Resize(self.img_size + 32),
                transforms.CenterCrop(self.img_size),
                transforms.ToTensor(),
            ]
        )
        self.patch = torch.rand(3, PATCH_RES, PATCH_RES, device=self.device, requires_grad=True)
        self.opt = torch.optim.Adam([self.patch], lr=ATTACK_LR)
        self.sched = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, T_max=ATTACK_STEPS)
        self.out_dir = Path(OUTPUT_DIR)
        self.evo_dir = self.out_dir / "patch_evolution"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.evo_dir.mkdir(parents=True, exist_ok=True)

    def load_images(self) -> list:
        paths = sorted(
            p
            for ext in ("*.png", "*.jpg", "*.jpeg")
            for p in Path(DATASET).resolve().glob(f"**/{ext}")
            if p.is_file()
        )
        print(f"{len(paths)} images dans {DATASET}")
        imgs = []
        for p in tqdm(paths, desc="Filtrage"):
            img = self.tf(Image.open(p).convert("RGB"))
            with torch.no_grad():
                tokens = F.normalize(
                    self.model.get_intermediate_layers(img.unsqueeze(0).to(self.device), n=1)[0][
                        0, 1:
                    ],
                    dim=-1,
                )
                preds = self.clf(tokens).argmax(-1)
            if (preds == SOURCE_CLASS).sum() >= ATTACK_MIN_SOURCE_TOKENS:
                imgs.append(img)
        print(
            f"{len(imgs)}/{len(paths)} images avec "
            f"≥{ATTACK_MIN_SOURCE_TOKENS} tokens '{self.src_name}'"
        )
        return imgs

    def sample_positions(self, ref_preds: torch.Tensor) -> tuple[list, list, list]:
        x_min = int(self.img_size * PATCH_MIN_ROW_RATIO)
        n_p = self.grid * self.grid
        positions, sizes, valid = [], [], []
        for i in range(ATTACK_BATCH_SIZE):
            pred_flat = ref_preds[i].cpu().numpy()
            pred_flat = np.concatenate(
                [pred_flat, np.full(max(0, n_p - len(pred_flat)), pred_flat[-1])]
            )[:n_p]
            src_map = pred_flat.reshape(self.grid, self.grid) == SOURCE_CLASS
            found = False
            x = y = 0
            eff = PATCH_SIZE
            for _ in range(50):
                x = int(
                    torch.randint(x_min, max(x_min + 1, self.img_size - PATCH_SIZE), (1,)).item()
                )
                eff = compute_perspective_size(
                    x, self.img_size, PATCH_SIZE, PATCH_PERSPECTIVE_MIN_SCALE
                )
                yc = int((self.img_size - eff) * PATCH_Y_RATIO)
                jit = self.img_size // 12
                y = int(
                    torch.randint(
                        max(0, yc - jit), min(self.img_size - eff, yc + jit) + 1, (1,)
                    ).item()
                )
                r0, r1 = x // 16, min((x + eff - 1) // 16 + 1, self.grid)
                c0, c1 = y // 16, min((y + eff - 1) // 16 + 1, self.grid)
                if not src_map[r0:r1, c0:c1].any():
                    found = True
                    break
            positions.append((x, y))
            sizes.append(eff)
            valid.append(found)
        return positions, sizes, valid

    def train_step(self, imgs: list) -> tuple | None:
        idx = torch.randint(0, len(imgs), (ATTACK_BATCH_SIZE,))
        batch = torch.stack([imgs[i] for i in idx]).to(self.device)
        with torch.no_grad():
            ref_tokens = F.normalize(
                self.model.get_intermediate_layers(batch, n=1)[0][:, 1:], dim=-1
            )
            ref_preds = self.clf(ref_tokens).argmax(-1)
        source_mask = ref_preds == SOURCE_CLASS
        if source_mask.sum() < ATTACK_MIN_SOURCE_TOKENS:
            return None
        positions, sizes, valid = self.sample_positions(ref_preds)
        valid_t = torch.tensor(valid, device=self.device)
        source_mask = source_mask & valid_t.unsqueeze(1)
        if source_mask.sum() < ATTACK_MIN_SOURCE_TOKENS:
            return None
        patched = apply_patch(batch.detach(), self.patch, sizes, positions)
        adv_tokens = F.normalize(self.model.get_intermediate_layers(patched, n=1)[0][:, 1:], dim=-1)
        adv_logits = self.clf(adv_tokens)
        target_t = torch.full(
            (source_mask.sum(),),
            SOURCE_CLASS if self.untargeted else TARGET_CLASS,
            device=self.device,
        )
        loss = (-1 if self.untargeted else 1) * F.cross_entropy(adv_logits[source_mask], target_t)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        self.sched.step()
        self.patch.data.clamp_(0, 1)
        with torch.no_grad():
            adv_p = adv_logits[source_mask].argmax(-1)
            fr = (
                (adv_p != SOURCE_CLASS if self.untargeted else adv_p == TARGET_CLASS)
                .float()
                .mean()
                .item()
            )
        return loss, fr, ref_preds, adv_tokens.detach(), patched, positions, sizes

    def render_frame(
        self,
        step: int,
        loss: torch.Tensor,
        fr: float,
        best_fr: float,
        ref_preds: torch.Tensor,
        adv_tokens: torch.Tensor,
        patched: torch.Tensor,
        positions: list,
        sizes: list,
    ) -> bool:
        sz = VIZ_SIZE
        img_np = (np.clip(patched[0].detach().cpu().permute(1, 2, 0).numpy(), 0, 1) * 255).astype(
            np.uint8
        )
        img_d = cv2.cvtColor(cv2.resize(img_np, (sz, sz)), cv2.COLOR_RGB2BGR)
        s = sz / self.img_size
        px, py = positions[0]
        rps = int(sizes[0] * s)
        cv2.rectangle(
            img_d,
            (int(py * s), int(px * s)),
            (int(py * s) + rps, int(px * s) + rps),
            (0, 255, 0),
            2,
        )
        ref_vis = colorize_preds(ref_preds[0].cpu().numpy(), self.grid, sz)
        adv_vis = colorize_preds(self.clf(adv_tokens[0]).argmax(-1).cpu().numpy(), self.grid, sz)
        frame = np.hstack(
            [img_d, ref_vis, adv_vis, patch_to_img(self.patch, sz), create_legend(sz)]
        )
        for i, lbl in enumerate(["Image+Patch", "Seg Original", "Seg Attaqué", "Patch"]):
            cv2.putText(
                frame, lbl, (i * sz + 5, sz - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1
            )
        cv2.putText(
            frame,
            f"Step {step}/{ATTACK_STEPS} | Loss: {loss.item():.4f}"
            f" | FR: {fr:.0%} | Best: {best_fr:.0%}",
            (10, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )
        cv2.imshow("Patch Training", frame)
        return cv2.waitKey(1) & 0xFF == ord("q")

    def save_results(self, step: int, evo_steps: list, history: list) -> None:
        torch.save(self.patch.detach().cpu(), self.out_dir / "targeted_patch_final.pt")
        if not evo_steps or evo_steps[-1] != step:
            torch.save(self.patch.detach().cpu(), self.evo_dir / f"patch_step_{step:05d}.pt")
            evo_steps.append(step)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        ax1.plot(history)
        ax1.set_title(f"Fooling rate ({self.src_name}→{self.tgt_name})")
        ax1.set_ylim(0, 1)
        ax2.imshow(np.clip(self.patch.detach().cpu().permute(1, 2, 0).numpy(), 0, 1))
        ax2.set_title("Patch")
        ax2.axis("off")
        plt.tight_layout()
        plt.savefig(self.out_dir / "targeted_attack_results.png", dpi=150)
        print(f"Résultats → {self.out_dir}")
        make_evolution_video(
            self.evo_dir,
            evo_steps,
            history,
            ATTACK_STEPS,
            self.out_dir,
            self.src_name,
            self.tgt_name,
        )

    def run(self) -> None:
        imgs = self.load_images()
        if not imgs:
            print("Aucune image valide.")
            return
        history, evo_steps, best_fr = [], [], 0.0
        cv2.namedWindow("Patch Training", cv2.WINDOW_NORMAL)
        step = 1
        for step in range(1, ATTACK_STEPS + 1):
            result = self.train_step(imgs)
            if result is None:
                continue
            loss, fr, ref_preds, adv_tokens, patched, positions, sizes = result
            history.append(fr)
            if fr > best_fr:
                best_fr = fr
                torch.save(self.patch.detach().cpu(), self.out_dir / "targeted_patch_best.pt")
            if step % PATCH_SAVE_EVERY == 0:
                torch.save(self.patch.detach().cpu(), self.evo_dir / f"patch_step_{step:05d}.pt")
                evo_steps.append(step)
            if step % 50 == 0:
                print(
                    f"Step {step}/{ATTACK_STEPS} | Loss: {loss.item():.4f} "
                    f"| Fooling: {fr:.0%} | Best: {best_fr:.0%}"
                )
            if VIZ_EVERY > 0 and step % VIZ_EVERY == 0:
                if self.render_frame(
                    step, loss, fr, best_fr, ref_preds, adv_tokens, patched, positions, sizes
                ):
                    print("Interrompu.")
                    break
        cv2.destroyAllWindows()
        print(f"\nMeilleur fooling rate : {best_fr:.0%}")
        self.save_results(step, evo_steps, history)


def main() -> None:
    PatchAttack().run()


if __name__ == "__main__":
    main()
