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
from utils.viz import CLASS_NAMES, colorize_preds, create_legend, patch_to_img, compute_perspective_size
from utils.config import (
    DATASET, CLASSIFIER, SOURCE_CLASS, TARGET_CLASS, ATTACK_STEPS,
    PATCH_SIZE, PATCH_RES, PATCH_PERSPECTIVE_MIN_SCALE, PATCH_MIN_ROW_RATIO,
    ATTACK_LR, ATTACK_BATCH_SIZE, ATTACK_MIN_SOURCE_TOKENS, OUTPUT_DIR,
    VIZ_EVERY, VIZ_SIZE, PATCH_SAVE_EVERY, PATCH_Y_RATIO,
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

def _make_evolution_video(evo_dir: Path, evo_steps: list[int], history: list[float],
                          total_steps: int, out_dir: Path, src: str, tgt: str) -> None:
    """Generate a patch evolution video (MP4) and a summary grid (PNG) from saved snapshots."""
    sz: int = 256  # frame size for patch in video

    # --- MP4 via cv2 ---
    video_path: Path = out_dir / "patch_evolution.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps_evo: int = max(2, min(15, len(evo_steps) // 5 + 1))
    writer = cv2.VideoWriter(str(video_path), fourcc, fps_evo, (sz + 400, sz))
    if not writer.isOpened():
        writer = None

    frames_rgb: list[np.ndarray] = []  # for GIF fallback
    for s in evo_steps:
        pt_path: Path = evo_dir / f"patch_step_{s:05d}.pt"
        if not pt_path.exists():
            continue
        p: torch.Tensor = torch.load(pt_path, map_location='cpu', weights_only=True)
        patch_img: np.ndarray = patch_to_img(p, sz)  # BGR

        # Fooling rate at this step (history index = step-1)
        idx: int = min(s - 1, len(history) - 1)
        fr: float = history[idx] if idx >= 0 else 0.0

        # Side panel: step + fooling rate bar
        panel: np.ndarray = np.zeros((sz, 400, 3), dtype=np.uint8)
        panel[:] = 25
        cv2.putText(panel, f"Step {s}/{total_steps}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (220, 220, 220), 2)
        cv2.putText(panel, f"Attack: {src} -> {tgt}", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
        # Fooling rate bar
        bar_w: int = int(360 * fr)
        cv2.rectangle(panel, (10, 110), (370, 150), (60, 60, 60), -1)
        if bar_w > 0:
            cv2.rectangle(panel, (10, 110), (10 + bar_w, 150), (0, 200, 80), -1)
        cv2.putText(panel, f"Fooling: {fr:.0%}", (10, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 100), 2)

        frame: np.ndarray = np.hstack([patch_img, panel])
        frames_rgb.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if writer is not None:
            writer.write(frame)

    if writer is not None:
        writer.release()
        print(f"Evolution video: {video_path}")

    # --- PNG grid: up to 20 snapshots evenly spaced ---
    n_show: int = min(20, len(evo_steps))
    indices: list[int] = [int(i * (len(evo_steps) - 1) / max(1, n_show - 1)) for i in range(n_show)]
    cols: int = min(5, n_show)
    rows: int = (n_show + cols - 1) // cols
    grid_sz: int = 128
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.5))
    axes_flat = np.array(axes).flatten() if n_show > 1 else [axes]
    for ax_i, idx_i in enumerate(indices):
        s = evo_steps[idx_i]
        pt_path = evo_dir / f"patch_step_{s:05d}.pt"
        ax = axes_flat[ax_i]
        if pt_path.exists():
            p = torch.load(pt_path, map_location='cpu', weights_only=True)
            img_arr = np.clip(p.permute(1, 2, 0).numpy(), 0, 1)
            fr_val = history[min(s - 1, len(history) - 1)] if history else 0.0
            ax.imshow(img_arr)
            ax.set_title(f"Step {s}\nFR {fr_val:.0%}", fontsize=8)
        ax.axis('off')
    for ax in axes_flat[n_show:]:
        ax.axis('off')
    fig.suptitle(f"Patch evolution — {src} → {tgt}", fontsize=12)
    plt.tight_layout()
    grid_path: Path = out_dir / "patch_evolution_grid.png"
    plt.savefig(grid_path, dpi=150)
    plt.close(fig)
    print(f"Evolution grid:  {grid_path}")


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
    parser.add_argument('--min-row-ratio', type=float, default=PATCH_MIN_ROW_RATIO,
                        help='Min vertical position as fraction of image height (0=top, 1=bottom)')
    parser.add_argument('--patch-y-ratio', type=float, default=PATCH_Y_RATIO,
                        help='Horizontal patch center as fraction of image width (0=left, 1=right)')
    parser.add_argument('--min-source-tokens', type=int, default=ATTACK_MIN_SOURCE_TOKENS,
                        help='Min number of source-class tokens required to include an image in training')
    parser.add_argument('--lr', type=float, default=ATTACK_LR)
    parser.add_argument('--batch-size', type=int, default=ATTACK_BATCH_SIZE)
    parser.add_argument('--output', default=OUTPUT_DIR)
    parser.add_argument('--viz-every', type=int, default=VIZ_EVERY)
    parser.add_argument('--viz-size', type=int, default=VIZ_SIZE)
    parser.add_argument('--save-every', type=int, default=PATCH_SAVE_EVERY,
                        help='Save patch snapshot every N steps for evolution replay (0=off)')
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
        if (preds == args.source_class).sum() >= args.min_source_tokens:
            imgs.append(img)
    print(f"{len(imgs)}/{len(paths)} images have ≥{args.min_source_tokens} '{src}' tokens")
    if len(imgs) == 0:
        print(f"No images contain class '{src}'. Exiting.")
        return

    patch: torch.Tensor = torch.rand(3, args.patch_res, args.patch_res, device=device, requires_grad=True)
    opt: torch.optim.Adam = torch.optim.Adam([patch], lr=args.lr)
    sched: torch.optim.lr_scheduler.CosineAnnealingLR = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)

    out_dir: Path = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    evo_dir: Path = out_dir / "patch_evolution"
    if args.save_every > 0:
        evo_dir.mkdir(parents=True, exist_ok=True)
    history: list[float] = []
    evo_steps: list[int] = []      # steps at which snapshots were saved
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

        # Reference predictions first — needed to avoid placing patch on source class
        with torch.no_grad():
            ref_tokens: torch.Tensor = F.normalize(model.get_intermediate_layers(batch, n=1)[0][:, 1:], dim=-1)
            ref_preds: torch.Tensor = clf(ref_tokens).argmax(-1)  # [B, grid*grid]
        source_mask: torch.Tensor = ref_preds == args.source_class
        if source_mask.sum() < args.min_source_tokens: continue

        # Sample positions that do not overlap with source-class tokens (up to 20 retries)
        ps: int = args.patch_size
        x_min: int = int(img_size * args.min_row_ratio)
        positions: list[tuple[int, int]] = []
        effective_sizes: list[int] = []
        for i in range(args.batch_size):
            src_map: np.ndarray = (ref_preds[i].cpu().numpy().reshape(grid, grid) == args.source_class)
            x, y, eff = 0, 0, ps
            for _ in range(20):
                x = torch.randint(x_min, max(x_min + 1, img_size - ps), (1,)).item()
                eff = compute_perspective_size(x, img_size, ps, args.perspective_min_scale)
                # Fix horizontal position to road-side with small jitter (avoids left/right jumping)
                y_center = int((img_size - eff) * args.patch_y_ratio)
                y_jitter = img_size // 12
                y = torch.randint(
                    max(0, y_center - y_jitter),
                    min(img_size - eff, y_center + y_jitter) + 1,
                    (1,)).item()
                r0, r1 = x // 16, min((x + eff - 1) // 16 + 1, grid)
                c0, c1 = y // 16, min((y + eff - 1) // 16 + 1, grid)
                if not src_map[r0:r1, c0:c1].any():
                    break
            positions.append((x, y))
            effective_sizes.append(eff)

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

        if args.save_every > 0 and step % args.save_every == 0:
            torch.save(patch.detach().cpu(), evo_dir / f"patch_step_{step:05d}.pt")
            evo_steps.append(step)

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
    if args.save_every > 0 and (not evo_steps or evo_steps[-1] != step):
        torch.save(patch.detach().cpu(), evo_dir / f"patch_step_{step:05d}.pt")
        evo_steps.append(step)
    print(f"\nBest fooling rate: {best_fr:.0%}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(history); ax1.set_title(f"Fooling rate ({src}->{tgt})"); ax1.set_ylim(0, 1)
    ax2.imshow(np.clip(patch.detach().cpu().permute(1, 2, 0).numpy(), 0, 1)); ax2.set_title("Patch"); ax2.axis('off')
    plt.tight_layout()
    plt.savefig(out_dir / "targeted_attack_results.png", dpi=150)
    print(f"Saved to {out_dir}")

    # Generate patch evolution video + grid
    if args.save_every > 0 and len(evo_steps) > 0:
        _make_evolution_video(evo_dir, evo_steps, history, args.steps, out_dir, src, tgt)

if __name__ == "__main__":
    main()
