"""Targeted adversarial patch: make source_class tokens be classified as target_class.

Usage:
    python scripts/attack_classifier.py --dataset data/stuttgart_00 --classifier results/classifier.pt
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import argparse
from tqdm import tqdm
from models.dinov3_loader import load_dinov3
from utils.viz import CLASS_NAMES, colorize_preds, create_legend, patch_to_img
from utils.config import (
    DATASET, CLASSIFIER, SOURCE_CLASS, TARGET_CLASS, ATTACK_STEPS,
    PATCH_SIZE, PATCH_RES, PATCH_PERSPECTIVE_MIN_SCALE,
    ATTACK_LR, ATTACK_BATCH_SIZE, OUTPUT_DIR,
    VIZ_EVERY, VIZ_SIZE,
)

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

def compute_perspective_size(x: int, img_size: int, max_size: int, min_scale: float) -> int:
    """Scale patch based on vertical position: smaller when higher (farther from camera)."""
    t = min(1.0, (x + max_size / 2) / img_size)  # normalized center height [0=top, 1=bottom]
    scale = min_scale + (1.0 - min_scale) * t
    return max(1, int(max_size * scale))


def apply_patch(images: torch.Tensor, patch: torch.Tensor, patch_sizes: list[int], positions: list[tuple[int, int]]) -> torch.Tensor:
    """Apply patch to images with per-image sizes (supports perspective scaling)."""
    out: torch.Tensor = images.clone()
    for i in range(images.shape[0]):
        ps = patch_sizes[i]
        resized: torch.Tensor = F.interpolate(patch.unsqueeze(0), size=(ps, ps), mode='bilinear', align_corners=False)[0]
        x, y = positions[i]
        xe: int = min(x + ps, images.shape[2])
        ye: int = min(y + ps, images.shape[3])
        out[i, :, x:xe, y:ye] = resized[:, :xe-x, :ye-y]
    return out

def main() -> None:
    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default=DATASET)
    parser.add_argument('--classifier', default=CLASSIFIER)
    parser.add_argument('--source-class', type=int, default=SOURCE_CLASS)
    parser.add_argument('--target-class', type=int, default=TARGET_CLASS)
    parser.add_argument('--steps', type=int, default=ATTACK_STEPS)
    parser.add_argument('--patch-size', type=int, default=PATCH_SIZE)
    parser.add_argument('--patch-res', type=int, default=PATCH_RES)
    parser.add_argument('--perspective-min-scale', type=float, default=PATCH_PERSPECTIVE_MIN_SCALE,
                        help='Min patch scale at top of image (0=invisible, 1=no perspective)')
    parser.add_argument('--lr', type=float, default=ATTACK_LR)
    parser.add_argument('--batch-size', type=int, default=ATTACK_BATCH_SIZE)
    parser.add_argument('--output', default=OUTPUT_DIR)
    parser.add_argument('--viz-every', type=int, default=VIZ_EVERY)
    parser.add_argument('--viz-size', type=int, default=VIZ_SIZE)
    args: argparse.Namespace = parser.parse_args()

    device: torch.device = get_device()
    src: str = CLASS_NAMES[args.source_class]
    untargeted: bool = args.target_class == -1
    tgt: str = "any" if untargeted else CLASS_NAMES[args.target_class]
    print(f"Attack: {src} -> {tgt} | Device: {device} | Patch: {args.patch_res}x{args.patch_res} -> {args.patch_size}x{args.patch_size}px | Perspective: {args.perspective_min_scale:.0%}–100%")

    model: nn.Module = load_model(device)
    data: dict = torch.load(args.classifier, map_location=device, weights_only=False)
    clf: nn.Linear = nn.Linear(384, 19).to(device)
    clf.load_state_dict(data['state_dict'])
    clf.eval()

    img_size: int = data.get('img_size', 224)
    print(f"Image size: {img_size}x{img_size} -> {img_size//16}x{img_size//16} tokens")

    tf: transforms.Compose = transforms.Compose([transforms.Resize(img_size + 32), transforms.CenterCrop(img_size), transforms.ToTensor()])
    paths: list[Path] = sorted(
        p for ext in ['*.png', '*.jpg', '*.jpeg']
        for p in Path(args.dataset).resolve().glob(f'**/{ext}') if p.is_file()
    )
    print(f"{len(paths)} images found in {args.dataset}")

    # Load and pre-filter: keep only images containing the source class
    imgs: list[torch.Tensor] = []
    print(f"Loading & filtering images containing '{src}'...")
    for p in tqdm(paths, desc="Filtering"):
        img: torch.Tensor = tf(Image.open(p).convert('RGB'))
        with torch.no_grad():
            tokens: torch.Tensor = F.normalize(model.get_intermediate_layers(img.unsqueeze(0).to(device), n=1)[0][0, 1:], dim=-1)
            preds: torch.Tensor = clf(tokens).argmax(-1)
        if (preds == args.source_class).sum() > 0:
            imgs.append(img)
    print(f"{len(imgs)}/{len(paths)} images contain '{src}'")
    if len(imgs) == 0:
        print(f"No images contain class '{src}'. Exiting.")
        return

    patch: torch.Tensor = torch.rand(3, args.patch_res, args.patch_res, device=device, requires_grad=True)
    opt: torch.optim.Adam = torch.optim.Adam([patch], lr=args.lr)
    sched: torch.optim.lr_scheduler.CosineAnnealingLR = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)

    out_dir: Path = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    history: list[float] = []
    best_fr: float = 0.0
    grid: int = img_size // 16
    stop_training: bool = False

    if args.viz_every > 0:
        cv2.namedWindow("Patch Training", cv2.WINDOW_NORMAL)
        cv2.moveWindow("Patch Training", 100, 100)

    for step in range(1, args.steps + 1):
        if stop_training:
            break
        idx: torch.Tensor = torch.randint(0, len(imgs), (args.batch_size,))
        batch: torch.Tensor = torch.stack([imgs[i] for i in idx]).to(device)
        ps: int = args.patch_size
        positions: list[tuple[int, int]] = []
        effective_sizes: list[int] = []
        for _ in range(args.batch_size):
            x: int = torch.randint(0, max(1, img_size - ps), (1,)).item()
            eff: int = compute_perspective_size(x, img_size, ps, args.perspective_min_scale)
            y: int = torch.randint(0, max(1, img_size - eff), (1,)).item()
            positions.append((x, y))
            effective_sizes.append(eff)

        with torch.no_grad():
            ref_tokens: torch.Tensor = F.normalize(model.get_intermediate_layers(batch, n=1)[0][:, 1:], dim=-1)
            ref_preds: torch.Tensor = clf(ref_tokens).argmax(-1)
        source_mask: torch.Tensor = ref_preds == args.source_class
        if source_mask.sum() == 0: continue

        patched: torch.Tensor = apply_patch(batch.detach(), patch, effective_sizes, positions)
        adv_tokens: torch.Tensor = F.normalize(model.get_intermediate_layers(patched, n=1)[0][:, 1:], dim=-1)
        adv_logits: torch.Tensor = clf(adv_tokens)

        if untargeted:
            loss = -F.cross_entropy(
                adv_logits[source_mask],
                torch.full((source_mask.sum(),), args.source_class, device=device)
            )
        else:
            loss = F.cross_entropy(
                adv_logits[source_mask],
                torch.full((source_mask.sum(),), args.target_class, device=device)
            )
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        patch.data.clamp_(0, 1)

        with torch.no_grad():
            adv_preds: torch.Tensor = adv_logits[source_mask].argmax(-1)
            if untargeted:
                fr: float = (adv_preds != args.source_class).float().mean().item()
            else:
                fr = (adv_preds == args.target_class).float().mean().item()
        history.append(fr)

        if fr > best_fr:
            best_fr = fr
            torch.save(patch.detach().cpu(), out_dir / "targeted_patch_best.pt")

        if step % 50 == 0:
            print(f"Step {step}/{args.steps} | Loss: {loss.item():.4f} | Fooling: {fr:.0%} | Tokens: {source_mask.sum()}")

        # Live visualization
        if args.viz_every > 0 and step % args.viz_every == 0:
            sz: int = args.viz_size
            # First image of batch with patch overlay
            img_np: np.ndarray = patched[0].detach().cpu().permute(1, 2, 0).numpy()
            img_np = (np.clip(img_np, 0, 1) * 255).astype(np.uint8)
            img_display: np.ndarray = cv2.resize(img_np, (sz, sz))
            img_display = cv2.cvtColor(img_display, cv2.COLOR_RGB2BGR)
            # Draw patch rectangle (perspective-scaled size)
            scale_viz: float = sz / img_size
            px, py = positions[0]
            rx, ry = int(py * scale_viz), int(px * scale_viz)
            rps = int(effective_sizes[0] * scale_viz)
            cv2.rectangle(img_display, (rx, ry), (rx + rps, ry + rps), (0, 255, 0), 2)

            # Segmentation of first image: ref vs adv
            ref_p: np.ndarray = ref_preds[0].cpu().numpy()
            adv_p: np.ndarray = clf(adv_tokens[0]).argmax(-1).cpu().numpy()
            ref_vis: np.ndarray = colorize_preds(ref_p, grid, sz)
            adv_vis: np.ndarray = colorize_preds(adv_p, grid, sz)

            # Patch enlarged
            patch_vis: np.ndarray = patch_to_img(patch, sz)

            legend_vis: np.ndarray = create_legend(sz)
            frame: np.ndarray = np.hstack([img_display, ref_vis, adv_vis, patch_vis, legend_vis])
            # Labels
            labels = ["Image+Patch", "Seg Original", "Seg Attacked", "Patch"]
            for i, label in enumerate(labels):
                cv2.putText(frame, label, (i * sz + 5, sz - 8),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            cv2.putText(frame, f"Step {step}/{args.steps} | Loss: {loss.item():.4f} | FR: {fr:.0%} | Best: {best_fr:.0%}",
                       (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            cv2.imshow("Patch Training", frame)
            key: int = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Training interrupted by user.")
                stop_training = True

    cv2.destroyAllWindows()
    torch.save(patch.detach().cpu(), out_dir / "targeted_patch_final.pt")
    print(f"\nBest fooling rate: {best_fr:.0%}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(history); ax1.set_title(f"Fooling rate ({src}->{tgt})"); ax1.set_ylim(0, 1)
    ax2.imshow(np.clip(patch.detach().cpu().permute(1, 2, 0).numpy(), 0, 1)); ax2.set_title("Patch"); ax2.axis('off')
    plt.tight_layout()
    plt.savefig(out_dir / "targeted_attack_results.png", dpi=150)
    print(f"Saved to {out_dir}")

if __name__ == "__main__":
    main()
