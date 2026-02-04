#!/usr/bin/env python3
"""
Entraînement d'un patch adversarial sur un dataset complet.
Génère un patch universel robuste.

Usage:
    # Sur le dataset Birds
    python scripts/train_dataset.py --dataset data/Birds --steps 2000

    # Avec augmentations EOT
    python scripts/train_dataset.py --dataset data/Birds --eot --steps 1500

    # Reprendre un entraînement
    python scripts/train_dataset.py --dataset data/Birds --resume results/patch.pt
"""

import os
# Enable MPS fallback for unsupported ops (grid_sampler_2d used by kornia)
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from torch.utils.data import Dataset, DataLoader
import argparse
import time
from tqdm import tqdm


# =========================
# Dataset
# =========================

class ImageDataset(Dataset):
    """Dataset d'images pour l'entraînement du patch."""

    def __init__(self, root_dir: str, transform=None, max_images: int = None):
        self.root_dir = Path(root_dir).resolve()  # Chemin absolu
        self.transform = transform

        # Trouver toutes les images (exclure les répertoires)
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.webp']
        self.image_paths = []
        for ext in extensions:
            for p in self.root_dir.glob(f'**/{ext}'):
                if p.is_file():
                    self.image_paths.append(p.resolve())  # Chemin absolu

        if max_images:
            self.image_paths = self.image_paths[:max_images]

        print(f"Found {len(self.image_paths)} images in {root_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image


# =========================
# Modèle
# =========================

def load_model(device=None):
    if device is None:
        device = (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
    print(f"Device: {device}")

    dinov3_path = Path(__file__).parent.parent / "src" / "models" / "dinov3"
    weights_path = Path(__file__).parent.parent / "src" / "models" / "weights" / "dinov3_vits16_pretrain_lvd1689m-08c60483.pth"

    model = torch.hub.load(str(dinov3_path), "dinov3_vits16", source="local", pretrained=False)

    if weights_path.exists():
        checkpoint = torch.load(str(weights_path), map_location=device, weights_only=False)
        model.load_state_dict(checkpoint, strict=True)
        print("Weights loaded")

    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad = False

    return model, device


# =========================
# Entraînement
# =========================

def apply_patch(images: torch.Tensor, patch: torch.Tensor, positions: list) -> torch.Tensor:
    """Applique le patch sur un batch d'images."""
    B = images.shape[0]
    patched = images.clone()
    ph, pw = patch.shape[1], patch.shape[2]

    for i in range(B):
        x, y = positions[i]
        x = max(0, min(x, images.shape[2] - ph))
        y = max(0, min(y, images.shape[3] - pw))
        patched[i, :, x:x+ph, y:y+pw] = patch

    return patched


def random_positions(batch_size: int, patch_size: int, img_size: int = 224) -> list:
    """Génère des positions aléatoires pour le patch."""
    max_pos = img_size - patch_size
    positions = []
    for _ in range(batch_size):
        x = np.random.randint(0, max_pos + 1)
        y = np.random.randint(0, max_pos + 1)
        positions.append((x, y))
    return positions


def apply_eot_transforms(patch: torch.Tensor, device: str) -> torch.Tensor:
    """Applique des transformations EOT au patch."""
    # Rotation aléatoire
    angle = torch.empty(1).uniform_(-30, 30).item()

    # Échelle aléatoire
    scale = torch.empty(1).uniform_(0.8, 1.2).item()

    # Brightness/contrast
    brightness = torch.empty(1).uniform_(-0.1, 0.1).to(device)
    contrast = torch.empty(1).uniform_(0.9, 1.1).to(device)

    # Appliquer
    transformed = patch.clone()

    # Rotation (simple, sans kornia pour éviter dépendances)
    if abs(angle) > 5:
        # Utiliser grid_sample pour rotation
        theta = torch.tensor([
            [np.cos(np.radians(angle)), -np.sin(np.radians(angle)), 0],
            [np.sin(np.radians(angle)), np.cos(np.radians(angle)), 0]
        ], dtype=torch.float32, device=device).unsqueeze(0)

        grid = F.affine_grid(theta, transformed.unsqueeze(0).size(), align_corners=False)
        transformed = F.grid_sample(transformed.unsqueeze(0), grid, align_corners=False).squeeze(0)

    # Brightness/contrast
    mean = transformed.mean()
    transformed = (transformed - mean) * contrast + mean + brightness

    return transformed.clamp(0, 1)


def train_on_dataset(
    model,
    dataloader: DataLoader,
    device: str,
    patch_size: int = 32,
    steps: int = 2000,
    lr: float = 0.05,
    use_eot: bool = False,
    n_eot: int = 4,
    resume_patch: torch.Tensor = None,
    save_every: int = 500,
    output_dir: str = "results",
) -> dict:
    """
    Entraîne un patch universel sur un dataset.

    Args:
        model: Modèle DINOv3
        dataloader: DataLoader du dataset
        device: Device
        patch_size: Taille du patch
        steps: Nombre de steps
        lr: Learning rate
        use_eot: Utiliser EOT
        n_eot: Nombre de transformations EOT par step
        resume_patch: Patch existant pour reprendre
        save_every: Sauvegarder tous les N steps
        output_dir: Dossier de sortie
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    print(f"\n{'='*50}")
    print(f"Training universal patch on dataset")
    print(f"{'='*50}")
    print(f"  Patch size: {patch_size}x{patch_size}")
    print(f"  Steps: {steps}")
    print(f"  LR: {lr}")
    print(f"  EOT: {use_eot} (n={n_eot})")
    print(f"  Batch size: {dataloader.batch_size}")
    print(f"  Dataset size: {len(dataloader.dataset)}")
    print()

    # Initialiser ou reprendre le patch
    if resume_patch is not None:
        patch = resume_patch.clone().to(device)
        patch.requires_grad = True
        print("Resuming from existing patch")
    else:
        patch = torch.rand(3, patch_size, patch_size, device=device, requires_grad=True)

    optimizer = torch.optim.Adam([patch], lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)

    # Historique
    history = {'mse': [], 'step': []}
    best_mse = 0
    best_patch = patch.detach().clone()

    # Itérateur infini sur le dataset
    data_iter = iter(dataloader)

    start_time = time.time()

    pbar = tqdm(range(steps), desc="Training")
    for step in pbar:
        # Obtenir un batch
        try:
            images = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            images = next(data_iter)

        images = images.to(device)
        batch_size = images.shape[0]

        optimizer.zero_grad()
        total_loss = 0.0
        n_samples = 0

        # Positions aléatoires pour ce batch
        positions = random_positions(batch_size, patch_size)

        # Tokens de référence (sans patch)
        with torch.no_grad():
            ref_features = model.get_intermediate_layers(images, n=1)[0]
            ref_tokens = F.normalize(ref_features[:, 1:], dim=-1)

        if use_eot:
            # Avec EOT: plusieurs transformations par step
            for _ in range(n_eot):
                transformed_patch = apply_eot_transforms(patch, device)
                patched_images = apply_patch(images, transformed_patch, positions)

                adv_features = model.get_intermediate_layers(patched_images, n=1)[0]
                adv_tokens = F.normalize(adv_features[:, 1:], dim=-1)

                # Loss par image
                for b in range(batch_size):
                    loss = -F.mse_loss(adv_tokens[b], ref_tokens[b])
                    total_loss += loss
                    n_samples += 1
        else:
            # Sans EOT
            patched_images = apply_patch(images, patch, positions)

            adv_features = model.get_intermediate_layers(patched_images, n=1)[0]
            adv_tokens = F.normalize(adv_features[:, 1:], dim=-1)

            for b in range(batch_size):
                loss = -F.mse_loss(adv_tokens[b], ref_tokens[b])
                total_loss += loss
                n_samples += 1

        total_loss /= n_samples
        total_loss.backward()
        optimizer.step()
        scheduler.step()
        patch.data.clamp_(0, 1)

        mse = -total_loss.item()

        if mse > best_mse:
            best_mse = mse
            best_patch = patch.detach().clone()

        # Logging
        history['mse'].append(mse)
        history['step'].append(step)

        pbar.set_postfix({
            'MSE': f'{mse:.6f}',
            'Best': f'{best_mse:.6f}',
            'LR': f'{scheduler.get_last_lr()[0]:.5f}'
        })

        # Sauvegarder périodiquement
        if (step + 1) % save_every == 0:
            checkpoint_path = output_dir / f"patch_step{step+1}.pt"
            torch.save(patch.detach(), checkpoint_path)
            tqdm.write(f"  Checkpoint saved: {checkpoint_path}")

    elapsed = time.time() - start_time
    print(f"\n{'='*50}")
    print(f"Training complete!")
    print(f"  Time: {elapsed:.1f}s ({elapsed/steps*1000:.1f}ms/step)")
    print(f"  Best MSE: {best_mse:.6f}")
    print(f"{'='*50}")

    return {
        'patch': best_patch,
        'history': history,
        'best_mse': best_mse,
    }


# =========================
# Visualisation
# =========================

def visualize_results(result: dict, model, dataloader, device, output_path: str, n_clusters: int = 4):
    """Visualise les résultats de l'entraînement."""
    patch = result['patch']
    history = result['history']

    # Prendre quelques images du dataset
    images = next(iter(dataloader))[:4].to(device)

    # Appliquer le patch
    positions = [(50, 50)] * 4  # Position fixe pour la viz
    patched_images = apply_patch(images, patch, positions)

    fig = plt.figure(figsize=(16, 12))

    # Row 1: Images originales vs avec patch
    for i in range(4):
        ax = fig.add_subplot(3, 4, i + 1)
        img = images[i].cpu().permute(1, 2, 0).numpy()
        ax.imshow(img)
        ax.set_title(f"Original {i+1}")
        ax.axis('off')

    for i in range(4):
        ax = fig.add_subplot(3, 4, i + 5)
        img = patched_images[i].detach().cpu().permute(1, 2, 0).numpy()
        ax.imshow(np.clip(img, 0, 1))
        ax.set_title(f"With Patch {i+1}")
        ax.axis('off')

    # Row 3: Patch + training curve
    ax = fig.add_subplot(3, 4, 9)
    patch_img = patch.cpu().permute(1, 2, 0).numpy()
    ax.imshow(np.clip(patch_img, 0, 1))
    ax.set_title(f"Adversarial Patch\n{patch.shape[1]}x{patch.shape[2]}")
    ax.axis('off')

    ax = fig.add_subplot(3, 4, 10)
    ax.plot(history['step'], history['mse'], 'b-', linewidth=0.5)
    ax.set_xlabel('Step')
    ax.set_ylabel('MSE')
    ax.set_title(f"Training Curve\nFinal: {result['best_mse']:.6f}")
    ax.grid(True, alpha=0.3)

    # Segmentation comparison
    with torch.no_grad():
        ref_features = model.get_intermediate_layers(images[:1], n=1)[0]
        ref_tokens = F.normalize(ref_features[0, 1:], dim=-1)

        adv_features = model.get_intermediate_layers(patched_images[:1], n=1)[0]
        adv_tokens = F.normalize(adv_features[0, 1:], dim=-1)

    kmeans = KMeans(n_clusters=n_clusters, n_init=3, random_state=42)
    ref_np = ref_tokens.cpu().numpy()
    adv_np = adv_tokens.cpu().numpy()
    # Pad to 196 if needed (some models have 195 tokens)
    if len(ref_np) < 196:
        ref_np = np.vstack([ref_np, np.zeros((196 - len(ref_np), ref_np.shape[1]))])
        adv_np = np.vstack([adv_np, np.zeros((196 - len(adv_np), adv_np.shape[1]))])
    labels_ref = kmeans.fit_predict(ref_np[:196]).reshape(14, 14)
    labels_adv = kmeans.fit_predict(adv_np[:196]).reshape(14, 14)

    ax = fig.add_subplot(3, 4, 11)
    ax.imshow(np.kron(labels_ref, np.ones((16, 16))), cmap='tab10')
    ax.set_title("Segmentation (Original)")
    ax.axis('off')

    ax = fig.add_subplot(3, 4, 12)
    ax.imshow(np.kron(labels_adv, np.ones((16, 16))), cmap='tab10')
    ax.set_title("Segmentation (Attacked)")
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to {output_path}")
    plt.close()


# =========================
# Main
# =========================

def main():
    parser = argparse.ArgumentParser(
        description='Train adversarial patch on dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic training
  python scripts/train_dataset.py --dataset data/Birds

  # With EOT for robustness
  python scripts/train_dataset.py --dataset data/Birds --eot

  # Longer training, larger patch
  python scripts/train_dataset.py --dataset data/Birds --patch-size 32 --steps 3000

  # Resume training
  python scripts/train_dataset.py --dataset data/Birds --resume results/patch.pt
        """
    )

    parser.add_argument('--dataset', type=str, required=True,
                       help='Path to dataset directory')
    parser.add_argument('--patch-size', type=int, default=32,
                       help='Patch size (default 24)')
    parser.add_argument('--steps', type=int, default=2000,
                       help='Training steps (default 2000)')
    parser.add_argument('--lr', type=float, default=0.05,
                       help='Learning rate (default 0.05)')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size (default 8)')
    parser.add_argument('--max-images', type=int, default=None,
                       help='Max images to use (default: all)')

    parser.add_argument('--eot', action='store_true',
                       help='Use EOT augmentations')
    parser.add_argument('--n-eot', type=int, default=4,
                       help='Number of EOT transforms per step')

    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint')
    parser.add_argument('--output', type=str, default='results',
                       help='Output directory')
    parser.add_argument('--clusters', type=int, default=4,
                       help='Number of K-means clusters for segmentation visualization')
    parser.add_argument('--save-every', type=int, default=500,
                       help='Save checkpoint every N steps')

    args = parser.parse_args()

    # Load model
    model, device = load_model()

    # Transform
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    # Dataset
    dataset = ImageDataset(args.dataset, transform, args.max_images)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # MPS doesn't support multiprocessing well
        drop_last=True,
    )

    # Resume patch
    resume_patch = None
    if args.resume:
        resume_patch = torch.load(args.resume, map_location=device)
        print(f"Resuming from {args.resume}")

    # Train
    result = train_on_dataset(
        model, dataloader, device,
        patch_size=args.patch_size,
        steps=args.steps,
        lr=args.lr,
        use_eot=args.eot,
        n_eot=args.n_eot,
        resume_patch=resume_patch,
        save_every=args.save_every,
        output_dir=args.output,
    )

    # Save final patch
    final_path = Path(args.output) / "universal_patch_final.pt"
    torch.save(result['patch'], final_path)
    print(f"\nFinal patch saved to {final_path}")

    # Visualize
    viz_path = Path(args.output) / "training_results.png"
    visualize_results(result, model, dataloader, device, str(viz_path), n_clusters=args.clusters)


if __name__ == "__main__":
    main()
