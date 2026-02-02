"""
DINOv3 Video Processor
Traitement vidéo avec extraction de features DINOv3
Optimisé pour Mac M3 Max (MPS)
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from torchvision import transforms
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Tuple
import time


# =========================
# Configuration
# =========================

@dataclass
class VideoConfig:
    """Configuration pour le traitement vidéo."""
    batch_size: int = 4          # Frames par batch (ajuster selon VRAM)
    frame_skip: int = 1          # 1 = toutes les frames, 2 = 1 sur 2, etc.
    max_frames: Optional[int] = None  # Limite de frames (None = toutes)
    resize_short_side: int = 256
    crop_size: int = 224
    show_progress: bool = True


# =========================
# Chargement du modèle
# =========================

def load_dinov3_model(model_name: str = "dinov3_vits16", device: str = None):
    """Charge le modèle DINOv3 avec détection automatique du device."""
    if device is None:
        device = (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )

    print(f"Using device: {device}")

    dinov3_path = Path(__file__).parent / "models" / "dinov3"
    weights_path = Path(__file__).parent / "models" / "weights" / f"{model_name}_pretrain_lvd1689m-08c60483.pth"

    model = torch.hub.load(
        str(dinov3_path),
        model_name,
        source="local",
        pretrained=False
    )

    if weights_path.exists():
        checkpoint = torch.load(str(weights_path), map_location=device, weights_only=False)
        model.load_state_dict(checkpoint, strict=True)
        print(f"Loaded weights from {weights_path.name}")
    else:
        print(f"Warning: Weights not found at {weights_path}")

    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad = False

    return model, device


# =========================
# Prétraitement
# =========================

def get_transform(config: VideoConfig):
    """Crée le pipeline de transformation pour les frames."""
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(config.resize_short_side),
        transforms.CenterCrop(config.crop_size),
        transforms.ToTensor(),
    ])


# =========================
# Extraction de features
# =========================

def extract_patch_tokens(model, images: torch.Tensor) -> torch.Tensor:
    """
    Extrait les patch tokens pour un batch d'images.

    Args:
        model: Modèle DINOv3
        images: Tensor de shape (B, 3, H, W)

    Returns:
        Tokens de shape (B, N, D) où N = nb patches, D = dim embedding
    """
    with torch.no_grad():
        features = model.get_intermediate_layers(images, n=1)[0]
        tokens = features[:, 1:]  # Remove CLS token
        tokens = F.normalize(tokens, dim=-1)
    return tokens


def extract_cls_token(model, images: torch.Tensor) -> torch.Tensor:
    """Extrait le CLS token (représentation globale) pour un batch."""
    with torch.no_grad():
        features = model.get_intermediate_layers(images, n=1)[0]
        cls_token = features[:, 0]  # CLS token only
        cls_token = F.normalize(cls_token, dim=-1)
    return cls_token


# =========================
# Processeur Vidéo
# =========================

class VideoProcessor:
    """Processeur vidéo pour extraction de features DINOv3."""

    def __init__(self, model=None, device: str = None, config: VideoConfig = None):
        if model is None:
            self.model, self.device = load_dinov3_model(device=device)
        else:
            self.model = model
            self.device = device or next(model.parameters()).device

        self.config = config or VideoConfig()
        self.transform = get_transform(self.config)
        self.patch_size = getattr(self.model, 'patch_size', 16)

    def process_video(self, video_path: str) -> dict:
        """
        Traite une vidéo et extrait les features.

        Args:
            video_path: Chemin vers la vidéo

        Returns:
            Dict contenant:
                - patch_tokens: List[Tensor] - tokens par frame
                - cls_tokens: Tensor (N_frames, D)
                - fps: float
                - frame_indices: List[int] - indices des frames traitées
                - processing_time: float
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"Video: {Path(video_path).name}")
        print(f"  FPS: {fps:.2f}, Total frames: {total_frames}")

        # Collecter les frames
        frames = []
        frame_indices = []
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Skip frames si configuré
            if frame_idx % self.config.frame_skip == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img_tensor = self.transform(frame_rgb)
                frames.append(img_tensor)
                frame_indices.append(frame_idx)

            frame_idx += 1

            # Limite de frames
            if self.config.max_frames and len(frames) >= self.config.max_frames:
                break

        cap.release()

        print(f"  Processing {len(frames)} frames...")

        # Traitement par batch
        start_time = time.time()
        all_patch_tokens = []
        all_cls_tokens = []

        for i in range(0, len(frames), self.config.batch_size):
            batch_end = min(i + self.config.batch_size, len(frames))
            batch = torch.stack(frames[i:batch_end]).to(self.device)

            patch_tokens = extract_patch_tokens(self.model, batch)
            cls_tokens = extract_cls_token(self.model, batch)

            all_patch_tokens.extend([t.cpu() for t in patch_tokens])
            all_cls_tokens.append(cls_tokens.cpu())

            if self.config.show_progress:
                progress = batch_end / len(frames) * 100
                print(f"\r  Progress: {progress:.1f}%", end="", flush=True)

        processing_time = time.time() - start_time

        if self.config.show_progress:
            print(f"\n  Done in {processing_time:.2f}s ({len(frames)/processing_time:.1f} FPS)")

        return {
            'patch_tokens': all_patch_tokens,
            'cls_tokens': torch.cat(all_cls_tokens, dim=0),
            'fps': fps,
            'frame_indices': frame_indices,
            'processing_time': processing_time,
            'num_frames': len(frames),
        }

    def compute_temporal_consistency(self, cls_tokens: torch.Tensor) -> np.ndarray:
        """
        Calcule la similarité cosinus entre frames consécutives.
        Utile pour détecter les changements de scène ou mouvements brusques.
        """
        similarities = F.cosine_similarity(
            cls_tokens[:-1], cls_tokens[1:], dim=1
        ).numpy()
        return similarities

    def find_keyframes(self, cls_tokens: torch.Tensor, threshold: float = 0.85) -> List[int]:
        """
        Trouve les keyframes basé sur les changements de représentation.

        Args:
            cls_tokens: CLS tokens de toutes les frames
            threshold: Seuil de similarité (frames sous ce seuil = keyframes)
        """
        similarities = self.compute_temporal_consistency(cls_tokens)
        keyframes = [0]  # Première frame toujours keyframe

        for i, sim in enumerate(similarities):
            if sim < threshold:
                keyframes.append(i + 1)

        return keyframes


# =========================
# Visualisation
# =========================

def visualize_video_features(results: dict, output_path: str = None):
    """Visualise les features extraites de la vidéo."""
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    cls_tokens = results['cls_tokens'].numpy()

    # PCA sur les CLS tokens
    pca = PCA(n_components=2)
    cls_2d = pca.fit_transform(cls_tokens)

    # Similarité temporelle
    processor = VideoProcessor.__new__(VideoProcessor)
    similarities = F.cosine_similarity(
        results['cls_tokens'][:-1],
        results['cls_tokens'][1:],
        dim=1
    ).numpy()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot 1: Trajectoire dans l'espace des features
    ax1 = axes[0]
    colors = np.arange(len(cls_2d))
    scatter = ax1.scatter(cls_2d[:, 0], cls_2d[:, 1], c=colors, cmap='viridis', s=20)
    ax1.plot(cls_2d[:, 0], cls_2d[:, 1], 'k-', alpha=0.3, linewidth=0.5)
    ax1.set_xlabel('PC1')
    ax1.set_ylabel('PC2')
    ax1.set_title('Feature Space Trajectory')
    plt.colorbar(scatter, ax=ax1, label='Frame')

    # Plot 2: Similarité temporelle
    ax2 = axes[1]
    ax2.plot(similarities, 'b-', linewidth=1)
    ax2.axhline(y=0.85, color='r', linestyle='--', label='Keyframe threshold')
    ax2.fill_between(range(len(similarities)), similarities, alpha=0.3)
    ax2.set_xlabel('Frame')
    ax2.set_ylabel('Cosine Similarity')
    ax2.set_title('Temporal Consistency')
    ax2.legend()
    ax2.set_ylim(0, 1)

    # Plot 3: Distribution des similarités
    ax3 = axes[2]
    ax3.hist(similarities, bins=50, color='purple', alpha=0.7, edgecolor='black')
    ax3.axvline(x=similarities.mean(), color='red', linestyle='--',
                label=f'Mean: {similarities.mean():.3f}')
    ax3.set_xlabel('Cosine Similarity')
    ax3.set_ylabel('Count')
    ax3.set_title('Similarity Distribution')
    ax3.legend()

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {output_path}")

    plt.show()

    return fig


# =========================
# Script principal
# =========================

def main():
    """Exemple d'utilisation du processeur vidéo."""
    import argparse

    parser = argparse.ArgumentParser(description='Process video with DINOv3')
    parser.add_argument('video_path', type=str, help='Path to input video')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
    parser.add_argument('--frame-skip', type=int, default=1, help='Process every Nth frame')
    parser.add_argument('--max-frames', type=int, default=None, help='Max frames to process')
    parser.add_argument('--output', type=str, default=None, help='Output path for visualization')
    parser.add_argument('--save-features', type=str, default=None, help='Save features to .pt file')

    args = parser.parse_args()

    # Configuration
    config = VideoConfig(
        batch_size=args.batch_size,
        frame_skip=args.frame_skip,
        max_frames=args.max_frames,
    )

    # Initialiser le processeur
    processor = VideoProcessor(config=config)

    # Traiter la vidéo
    results = processor.process_video(args.video_path)

    # Afficher les stats
    print(f"\nResults:")
    print(f"  Frames processed: {results['num_frames']}")
    print(f"  CLS tokens shape: {results['cls_tokens'].shape}")
    print(f"  Patch tokens per frame: {results['patch_tokens'][0].shape}")

    # Keyframes
    keyframes = processor.find_keyframes(results['cls_tokens'])
    print(f"  Keyframes detected: {len(keyframes)}")

    # Sauvegarder les features
    if args.save_features:
        torch.save(results, args.save_features)
        print(f"\nFeatures saved to {args.save_features}")

    # Visualisation
    output_path = args.output or str(
        Path(__file__).parent.parent / "results" / "video_analysis.png"
    )
    visualize_video_features(results, output_path)


if __name__ == "__main__":
    main()
