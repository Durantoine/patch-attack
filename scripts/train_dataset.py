"""Train universal adversarial patch on a dataset.

Usage:
    python scripts/train_dataset.py --dataset data/stuttgart_00 --steps 2000
    python scripts/train_dataset.py --dataset data/stuttgart_00 --eot --steps 1500
    python scripts/train_dataset.py --dataset data/stuttgart_00 --resume results/patch.pt
"""
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from sklearn.cluster import KMeans
from torch.utils.data import Dataset, DataLoader
import argparse
import time
from tqdm import tqdm
from models.dinov3_loader import load_dinov3

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

class ImageDataset(Dataset):
    def __init__(self, root_dir: str, transform: transforms.Compose | None = None, max_images: int | None = None):
        self.transform: transforms.Compose | None = transform
        root: Path = Path(root_dir).resolve()
        self.paths: list[Path] = sorted(
            p for ext in ['*.jpg', '*.jpeg', '*.png', '*.webp']
            for p in root.glob(f'**/{ext}') if p.is_file()
        )
        if max_images:
            self.paths = self.paths[:max_images]
        print(f"{len(self.paths)} images in {root_dir}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img: Image.Image = Image.open(self.paths[idx]).convert('RGB')
        return self.transform(img) if self.transform else transforms.ToTensor()(img)

def apply_patch(images: torch.Tensor, patch: torch.Tensor, positions: list[tuple[int, int]]) -> torch.Tensor:
    out: torch.Tensor = images.clone()
    ph: int = patch.shape[1]
    pw: int = patch.shape[2]
    for i in range(images.shape[0]):
        x, y = positions[i]
        x, y = max(0, min(x, images.shape[2]-ph)), max(0, min(y, images.shape[3]-pw))
        out[i, :, x:x+ph, y:y+pw] = patch
    return out

def random_positions(batch_size: int, patch_size: int, img_size: int = 224) -> list[tuple[int, int]]:
    m: int = img_size - patch_size
    return [(np.random.randint(0, m+1), np.random.randint(0, m+1)) for _ in range(batch_size)]

def apply_eot(patch: torch.Tensor, device: torch.device) -> torch.Tensor:
    angle: float = torch.empty(1).uniform_(-30, 30).item()
    brightness: torch.Tensor = torch.empty(1).uniform_(-0.1, 0.1).to(device)
    contrast: torch.Tensor = torch.empty(1).uniform_(0.9, 1.1).to(device)
    t: torch.Tensor = patch.clone()
    if abs(angle) > 5:
        theta: torch.Tensor = torch.tensor([
            [np.cos(np.radians(angle)), -np.sin(np.radians(angle)), 0],
            [np.sin(np.radians(angle)), np.cos(np.radians(angle)), 0]
        ], dtype=torch.float32, device=device).unsqueeze(0)
        grid: torch.Tensor = F.affine_grid(theta, t.unsqueeze(0).size(), align_corners=False)
        t = F.grid_sample(t.unsqueeze(0), grid, align_corners=False).squeeze(0)
    mean: torch.Tensor = t.mean()
    t = (t - mean) * contrast + mean + brightness
    return t.clamp(0, 1)

def train(model: nn.Module, dataloader: DataLoader, device: torch.device,
          patch_size: int = 32, steps: int = 2000, lr: float = 0.05,
          use_eot: bool = False, n_eot: int = 4, resume_patch: torch.Tensor | None = None,
          save_every: int = 500, output_dir: str = "results") -> dict:
    out_dir: Path = Path(output_dir)
    out_dir.mkdir(exist_ok=True)

    if resume_patch is not None:
        patch: torch.Tensor = resume_patch.clone().to(device)
        patch.requires_grad = True
        print("Resuming from existing patch")
    else:
        patch = torch.rand(3, patch_size, patch_size, device=device, requires_grad=True)

    opt: torch.optim.Adam = torch.optim.Adam([patch], lr=lr)
    sched: torch.optim.lr_scheduler.CosineAnnealingLR = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)

    history: list[float] = []
    best_mse: float = 0.0
    best_patch: torch.Tensor = patch.detach().clone()
    data_iter: iter = iter(dataloader)
    start: float = time.time()

    for step in tqdm(range(steps), desc="Training"):
        try:
            images: torch.Tensor = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            images = next(data_iter)
        images = images.to(device)
        bs: int = images.shape[0]
        positions: list[tuple[int, int]] = random_positions(bs, patch_size)

        with torch.no_grad():
            ref_tokens: torch.Tensor = F.normalize(model.get_intermediate_layers(images, n=1)[0][:, 1:], dim=-1)

        opt.zero_grad()
        total_loss: torch.Tensor = torch.tensor(0.0, device=device)
        n: int = 0

        for _ in range(n_eot if use_eot else 1):
            p: torch.Tensor = apply_eot(patch, device) if use_eot else patch
            patched: torch.Tensor = apply_patch(images, p, positions)
            adv_tokens: torch.Tensor = F.normalize(model.get_intermediate_layers(patched, n=1)[0][:, 1:], dim=-1)
            for b in range(bs):
                total_loss += -F.mse_loss(adv_tokens[b], ref_tokens[b])
                n += 1

        (total_loss / n).backward()
        opt.step(); sched.step()
        patch.data.clamp_(0, 1)

        mse: float = -total_loss.item() / n
        history.append(mse)
        if mse > best_mse:
            best_mse = mse
            best_patch = patch.detach().clone()

        if (step + 1) % save_every == 0:
            torch.save(patch.detach(), out_dir / f"patch_step{step+1}.pt")

    elapsed: float = time.time() - start
    print(f"Done in {elapsed:.1f}s | Best MSE: {best_mse:.6f}")
    return {'patch': best_patch, 'history': history, 'best_mse': best_mse}

def visualize_results(result: dict, model: nn.Module, dataloader: DataLoader,
                      device: torch.device, output_path: str, n_clusters: int = 4) -> None:
    patch: torch.Tensor = result['patch']
    history: list[float] = result['history']
    images: torch.Tensor = next(iter(dataloader))[:4].to(device)
    positions: list[tuple[int, int]] = [(50, 50)] * 4
    patched: torch.Tensor = apply_patch(images, patch, positions)

    fig: plt.Figure = plt.figure(figsize=(16, 12))
    for i in range(4):
        fig.add_subplot(3, 4, i+1)
        plt.imshow(images[i].cpu().permute(1, 2, 0).numpy()); plt.title(f"Original {i+1}"); plt.axis('off')
        fig.add_subplot(3, 4, i+5)
        plt.imshow(np.clip(patched[i].detach().cpu().permute(1, 2, 0).numpy(), 0, 1)); plt.title(f"Patched {i+1}"); plt.axis('off')

    fig.add_subplot(3, 4, 9)
    plt.imshow(np.clip(patch.cpu().permute(1, 2, 0).numpy(), 0, 1)); plt.title("Patch"); plt.axis('off')
    fig.add_subplot(3, 4, 10)
    plt.plot(history, 'b-', linewidth=0.5); plt.xlabel('Step'); plt.ylabel('MSE')
    plt.title(f"MSE: {result['best_mse']:.6f}"); plt.grid(True, alpha=0.3)

    with torch.no_grad():
        ref: np.ndarray = F.normalize(model.get_intermediate_layers(images[:1], n=1)[0][0, 1:], dim=-1).cpu().numpy()
        adv: np.ndarray = F.normalize(model.get_intermediate_layers(patched[:1], n=1)[0][0, 1:], dim=-1).cpu().numpy()

    km: KMeans = KMeans(n_clusters=n_clusters, n_init=3, random_state=42)
    fig.add_subplot(3, 4, 11)
    plt.imshow(np.kron(km.fit_predict(ref[:196]).reshape(14, 14), np.ones((16, 16))), cmap='tab10')
    plt.title("Seg Original"); plt.axis('off')
    fig.add_subplot(3, 4, 12)
    plt.imshow(np.kron(km.fit_predict(adv[:196]).reshape(14, 14), np.ones((16, 16))), cmap='tab10')
    plt.title("Seg Attacked"); plt.axis('off')

    plt.tight_layout(); plt.savefig(output_path, dpi=150, bbox_inches='tight'); plt.close()
    print(f"Visualization saved to {output_path}")

def main() -> None:
    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--patch-size', type=int, default=32)
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--max-images', type=int, default=None)
    parser.add_argument('--eot', action='store_true')
    parser.add_argument('--n-eot', type=int, default=4)
    parser.add_argument('--resume', default=None)
    parser.add_argument('--output', default='results')
    parser.add_argument('--clusters', type=int, default=4)
    parser.add_argument('--save-every', type=int, default=500)
    args: argparse.Namespace = parser.parse_args()

    device: torch.device = get_device()
    print(f"Device: {device}")
    model: nn.Module = load_model(device)

    tf: transforms.Compose = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()])
    dataset: ImageDataset = ImageDataset(args.dataset, tf, args.max_images)
    loader: DataLoader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)

    resume_patch: torch.Tensor | None = torch.load(args.resume, map_location=device, weights_only=False) if args.resume else None

    result: dict = train(model, loader, device, patch_size=args.patch_size, steps=args.steps,
                         lr=args.lr, use_eot=args.eot, n_eot=args.n_eot, resume_patch=resume_patch,
                         save_every=args.save_every, output_dir=args.output)

    torch.save(result['patch'], Path(args.output) / "universal_patch_final.pt")
    print(f"Patch saved to {args.output}/universal_patch_final.pt")
    visualize_results(result, model, loader, device, str(Path(args.output) / "training_results.png"), n_clusters=args.clusters)

if __name__ == "__main__":
    main()
