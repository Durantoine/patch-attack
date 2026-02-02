#!/usr/bin/env python3
"""
Script d'entraînement de patch adversarial
Génère un patch de haute qualité (comme test2.py mais configurable)

Usage:
    # Entraîner sur une image
    python scripts/train_patch.py --image data/Birds/amazon/image.jpg

    # Entraîner sur la webcam (capture une frame)
    python scripts/train_patch.py --webcam

    # Entraîner sur plusieurs images (patch universel)
    python scripts/train_patch.py --images-dir data/Birds/

    # Charger et continuer l'entraînement
    python scripts/train_patch.py --image data/image.jpg --resume results/patch.pt
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import argparse
import time


# =========================
# Chargement modèle
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
# Fonctions d'attaque
# =========================

def get_patch_tokens(model, image: torch.Tensor) -> torch.Tensor:
    """Extrait les patch tokens sans gradients."""
    with torch.no_grad():
        features = model.get_intermediate_layers(image.unsqueeze(0), n=1)[0]
    tokens = features[0, 1:]
    return F.normalize(tokens, dim=1)


def get_patch_tokens_diff(model, image: torch.Tensor) -> torch.Tensor:
    """Extrait les patch tokens avec gradients."""
    features = model.get_intermediate_layers(image.unsqueeze(0), n=1)[0]
    tokens = features[0, 1:]
    return F.normalize(tokens, dim=1)


def apply_patch(image: torch.Tensor, patch: torch.Tensor, pos: tuple) -> torch.Tensor:
    """Applique le patch sur l'image."""
    x, y = pos
    patched = image.clone()
    patched[:, x:x+patch.shape[1], y:y+patch.shape[2]] = patch
    return patched


def train_single_image(
    model,
    image: torch.Tensor,
    device: str,
    patch_size: int = 16,
    patch_pos: tuple = (0, 0),
    steps: int = 1000,
    lr: float = 0.05,
    resume_patch: torch.Tensor = None,
) -> dict:
    """
    Entraîne un patch sur une seule image (comme test2.py).

    Returns:
        dict avec patch, metrics, etc.
    """
    print(f"\n=== Training on single image ===")
    print(f"  Patch size: {patch_size}x{patch_size}")
    print(f"  Position: {patch_pos}")
    print(f"  Steps: {steps}")
    print(f"  LR: {lr}")

    # Initialiser ou reprendre le patch
    if resume_patch is not None:
        patch = resume_patch.clone().to(device)
        patch.requires_grad = True
        print(f"  Resuming from existing patch")
    else:
        patch = torch.rand(3, patch_size, patch_size, device=device, requires_grad=True)

    optimizer = torch.optim.Adam([patch], lr=lr)

    # Tokens de référence
    tokens_ref = get_patch_tokens(model, image)

    # Historique
    history = {'mse': [], 'cosine': []}
    best_mse = 0
    best_patch = patch.clone()

    start_time = time.time()

    for step in range(steps):
        optimizer.zero_grad()

        # Appliquer le patch
        patched_img = apply_patch(image, patch, patch_pos)

        # Forward avec gradients
        tokens_adv = get_patch_tokens_diff(model, patched_img)

        # Loss: maximiser la distance
        loss = -F.mse_loss(tokens_adv, tokens_ref)
        loss.backward()
        optimizer.step()

        # Clamp
        patch.data.clamp_(0, 1)

        # Metrics
        with torch.no_grad():
            mse = F.mse_loss(tokens_adv, tokens_ref).item()
            cosine = F.cosine_similarity(tokens_adv, tokens_ref, dim=1).mean().item()

        history['mse'].append(mse)
        history['cosine'].append(cosine)

        if mse > best_mse:
            best_mse = mse
            best_patch = patch.detach().clone()

        if step % 100 == 0:
            elapsed = time.time() - start_time
            print(f"  [{step:4d}/{steps}] MSE: {mse:.6f} | Cosine: {cosine:.4f} | Time: {elapsed:.1f}s")

    elapsed = time.time() - start_time
    print(f"\n  Done! Best MSE: {best_mse:.6f} in {elapsed:.1f}s")

    return {
        'patch': best_patch,
        'history': history,
        'final_mse': best_mse,
        'tokens_ref': tokens_ref,
        'image': image,
        'patch_pos': patch_pos,
    }


def train_universal(
    model,
    images: list,
    device: str,
    patch_size: int = 32,
    steps: int = 800,
    lr: float = 0.05,
    samples_per_step: int = 4,
) -> dict:
    """
    Entraîne un patch universel sur plusieurs images.
    """
    print(f"\n=== Training universal patch ===")
    print(f"  Images: {len(images)}")
    print(f"  Patch size: {patch_size}x{patch_size}")
    print(f"  Steps: {steps}")
    print(f"  LR: {lr}")
    print(f"  Samples/step: {samples_per_step}")

    patch = torch.rand(3, patch_size, patch_size, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([patch], lr=lr)

    # Position centrale
    center = 112 - patch_size // 2
    pos = (center, center)

    # Tokens de référence pour toutes les images
    ref_tokens = []
    for img in images:
        tokens = get_patch_tokens(model, img)
        ref_tokens.append(tokens)

    history = {'mse': []}
    best_mse = 0
    best_patch = patch.clone()

    start_time = time.time()

    for step in range(steps):
        optimizer.zero_grad()
        total_loss = 0.0

        # Sample quelques images
        indices = np.random.choice(len(images), min(samples_per_step, len(images)), replace=False)

        for idx in indices:
            patched_img = apply_patch(images[idx], patch, pos)
            tokens_adv = get_patch_tokens_diff(model, patched_img)
            loss = -F.mse_loss(tokens_adv, ref_tokens[idx])
            total_loss += loss

        total_loss /= len(indices)
        total_loss.backward()
        optimizer.step()
        patch.data.clamp_(0, 1)

        mse = -total_loss.item()
        history['mse'].append(mse)

        if mse > best_mse:
            best_mse = mse
            best_patch = patch.detach().clone()

        if step % 100 == 0:
            elapsed = time.time() - start_time
            print(f"  [{step:4d}/{steps}] MSE: {mse:.6f} | Time: {elapsed:.1f}s")

    print(f"\n  Done! Best MSE: {best_mse:.6f}")

    return {
        'patch': best_patch,
        'history': history,
        'final_mse': best_mse,
    }


# =========================
# Visualisation
# =========================

def visualize_attack(result: dict, model, device, output_path: str):
    """Génère une visualisation complète de l'attaque."""
    patch = result['patch']
    image = result['image']
    pos = result['patch_pos']

    # Appliquer le patch
    patched_img = apply_patch(image, patch, pos)

    # Tokens
    tokens_ref = get_patch_tokens(model, image)
    tokens_adv = get_patch_tokens(model, patched_img)

    # PCA
    all_tokens = torch.cat([tokens_ref, tokens_adv], dim=0).cpu().numpy()
    pca = PCA(n_components=2)
    pca.fit(all_tokens)
    ref_2d = pca.transform(tokens_ref.cpu().numpy())
    adv_2d = pca.transform(tokens_adv.cpu().numpy())

    # Distances
    token_dist = torch.norm(tokens_adv - tokens_ref, dim=1).cpu().numpy()

    # Segmentation
    kmeans = KMeans(n_clusters=4, n_init=3, random_state=42)
    labels_ref = kmeans.fit_predict(tokens_ref.cpu().numpy())
    labels_adv = kmeans.fit_predict(tokens_adv.cpu().numpy())

    # Plot
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 4, hspace=0.3, wspace=0.3)

    # Row 1
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(image.cpu().permute(1, 2, 0).numpy())
    ax1.set_title("Original", fontsize=12, fontweight='bold')
    ax1.axis('off')

    ax2 = fig.add_subplot(gs[0, 1])
    patched_display = patched_img.detach().cpu().permute(1, 2, 0).numpy()
    ax2.imshow(np.clip(patched_display, 0, 1))
    ax2.set_title(f"With Patch ({patch.shape[1]}x{patch.shape[2]})", fontsize=12, fontweight='bold')
    ax2.axis('off')

    ax3 = fig.add_subplot(gs[0, 2])
    patch_display = patch.detach().cpu().permute(1, 2, 0).numpy()
    ax3.imshow(np.clip(patch_display, 0, 1))
    ax3.set_title("Adversarial Patch", fontsize=12, fontweight='bold')
    ax3.axis('off')

    ax4 = fig.add_subplot(gs[0, 3])
    ax4.scatter(ref_2d[:, 0], ref_2d[:, 1], c='blue', alpha=0.5, s=20, label='Original')
    ax4.scatter(adv_2d[:, 0], adv_2d[:, 1], c='red', alpha=0.5, s=20, label='Attacked')
    ax4.legend()
    ax4.set_title("Token Space (PCA)", fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)

    # Row 2
    ax5 = fig.add_subplot(gs[1, 0])
    seg_ref = labels_ref.reshape(14, 14)
    ax5.imshow(np.kron(seg_ref, np.ones((16, 16))), cmap='tab10')
    ax5.set_title("Segmentation (Original)", fontsize=12, fontweight='bold')
    ax5.axis('off')

    ax6 = fig.add_subplot(gs[1, 1])
    seg_adv = labels_adv.reshape(14, 14)
    ax6.imshow(np.kron(seg_adv, np.ones((16, 16))), cmap='tab10')
    ax6.set_title("Segmentation (Attacked)", fontsize=12, fontweight='bold')
    ax6.axis('off')

    ax7 = fig.add_subplot(gs[1, 2])
    dist_map = token_dist.reshape(14, 14)
    im = ax7.imshow(np.kron(dist_map, np.ones((16, 16))), cmap='hot')
    plt.colorbar(im, ax=ax7, fraction=0.046)
    ax7.set_title(f"Token Distance (avg={token_dist.mean():.4f})", fontsize=12, fontweight='bold')
    ax7.axis('off')

    ax8 = fig.add_subplot(gs[1, 3])
    ax8.plot(result['history']['mse'], 'b-', linewidth=1)
    ax8.set_xlabel('Step')
    ax8.set_ylabel('MSE')
    ax8.set_title(f"Training (final={result['final_mse']:.6f})", fontsize=12, fontweight='bold')
    ax8.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to {output_path}")
    plt.close()


# =========================
# Main
# =========================

def main():
    parser = argparse.ArgumentParser(description='Train adversarial patch')

    # Source
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--image', type=str, help='Path to image')
    source.add_argument('--webcam', action='store_true', help='Capture from webcam')
    source.add_argument('--images-dir', type=str, help='Directory of images (universal patch)')

    # Patch params
    parser.add_argument('--patch-size', type=int, default=16, help='Patch size (default 16)')
    parser.add_argument('--patch-pos', type=int, nargs=2, default=[0, 0], help='Patch position x y')
    parser.add_argument('--steps', type=int, default=1000, help='Training steps')
    parser.add_argument('--lr', type=float, default=0.05, help='Learning rate')

    # Output
    parser.add_argument('--output', type=str, default=None, help='Output path for patch')
    parser.add_argument('--resume', type=str, default=None, help='Resume from existing patch')
    parser.add_argument('--no-viz', action='store_true', help='Skip visualization')

    args = parser.parse_args()

    # Load model
    model, device = load_model()

    # Transform
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    # Load image(s)
    if args.image:
        print(f"Loading image: {args.image}")
        image = Image.open(args.image).convert("RGB")
        image_tensor = transform(image).to(device)

        # Resume patch if provided
        resume_patch = None
        if args.resume:
            resume_patch = torch.load(args.resume, map_location=device)
            print(f"Resuming from {args.resume}")

        result = train_single_image(
            model, image_tensor, device,
            patch_size=args.patch_size,
            patch_pos=tuple(args.patch_pos),
            steps=args.steps,
            lr=args.lr,
            resume_patch=resume_patch,
        )

    elif args.webcam:
        print("Capturing from webcam...")
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            print("Failed to capture from webcam")
            return

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        image_tensor = transform(image).to(device)

        result = train_single_image(
            model, image_tensor, device,
            patch_size=args.patch_size,
            patch_pos=tuple(args.patch_pos),
            steps=args.steps,
            lr=args.lr,
        )

    else:  # images-dir
        print(f"Loading images from: {args.images_dir}")
        images_dir = Path(args.images_dir)
        image_paths = list(images_dir.glob("**/*.jpg")) + list(images_dir.glob("**/*.png"))
        print(f"Found {len(image_paths)} images")

        if len(image_paths) == 0:
            print("No images found!")
            return

        # Load max 50 images
        images = []
        for p in image_paths[:50]:
            img = Image.open(p).convert("RGB")
            img_tensor = transform(img).to(device)
            images.append(img_tensor)

        result = train_universal(
            model, images, device,
            patch_size=args.patch_size,
            steps=args.steps,
            lr=args.lr,
        )
        # For visualization, use first image
        result['image'] = images[0]
        result['patch_pos'] = (112 - args.patch_size // 2, 112 - args.patch_size // 2)
        result['tokens_ref'] = get_patch_tokens(model, images[0])

    # Save patch
    output_path = args.output or str(Path(__file__).parent.parent / "results" / "trained_patch.pt")
    Path(output_path).parent.mkdir(exist_ok=True)
    torch.save(result['patch'], output_path)
    print(f"\nPatch saved to {output_path}")

    # Visualization
    if not args.no_viz:
        viz_path = output_path.replace('.pt', '_viz.png')
        visualize_attack(result, model, device, viz_path)

    # Print summary
    print(f"\n=== Summary ===")
    print(f"  Patch size: {result['patch'].shape[1]}x{result['patch'].shape[2]}")
    print(f"  Final MSE: {result['final_mse']:.6f}")
    print(f"  Saved to: {output_path}")


if __name__ == "__main__":
    main()
