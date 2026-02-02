"""
Attaque adversariale par patch sur vidéo avec DINOv3
Optimisé pour Mac M3 Max (MPS)
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from torchvision import transforms
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple
import time

# Ajouter le parent au path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from video_processor import VideoProcessor, VideoConfig


@dataclass
class AttackConfig:
    """Configuration de l'attaque."""
    patch_size: int = 32           # Taille du patch adversarial
    patch_position: Tuple[int, int] = (0, 0)  # Position (x, y)
    steps: int = 500               # Itérations d'optimisation
    lr: float = 0.05               # Learning rate
    temporal_weight: float = 0.1   # Poids de la cohérence temporelle
    target_mode: str = "maximize_distance"  # ou "minimize_distance", "target_class"


class VideoAttacker:
    """Attaque adversariale sur vidéo."""

    def __init__(self, model=None, device: str = None, attack_config: AttackConfig = None):
        if model is None:
            processor = VideoProcessor()
            self.model = processor.model
            self.device = processor.device
        else:
            self.model = model
            self.device = device or str(next(model.parameters()).device)

        self.config = attack_config or AttackConfig()
        self.patch_size_model = getattr(self.model, 'patch_size', 16)

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])

    def _get_patch_tokens(self, images: torch.Tensor) -> torch.Tensor:
        """Extrait les patch tokens (avec gradients si nécessaire)."""
        features = self.model.get_intermediate_layers(images, n=1)[0]
        tokens = features[:, 1:]  # Remove CLS
        tokens = F.normalize(tokens, dim=-1)
        return tokens

    def _apply_patch(self, image: torch.Tensor, patch: torch.Tensor) -> torch.Tensor:
        """Applique le patch adversarial sur l'image."""
        x, y = self.config.patch_position
        patched = image.clone()

        # Gérer batch et single image
        if patched.dim() == 3:
            patched[:, x:x+patch.shape[1], y:y+patch.shape[2]] = patch
        else:
            patched[:, :, x:x+patch.shape[1], y:y+patch.shape[2]] = patch.unsqueeze(0)

        return patched

    def _attack_loss(self, tokens_ref: torch.Tensor, tokens_adv: torch.Tensor) -> torch.Tensor:
        """Calcule la loss d'attaque."""
        if self.config.target_mode == "maximize_distance":
            # Éloigner les représentations
            return -F.mse_loss(tokens_adv, tokens_ref)
        elif self.config.target_mode == "minimize_distance":
            # Rapprocher (pour des attaques ciblées)
            return F.mse_loss(tokens_adv, tokens_ref)
        else:
            return -F.mse_loss(tokens_adv, tokens_ref)

    def _temporal_consistency_loss(self, patches: list) -> torch.Tensor:
        """Pénalise les changements brusques du patch entre frames."""
        if len(patches) < 2:
            return torch.tensor(0.0, device=self.device)

        loss = 0.0
        for i in range(len(patches) - 1):
            loss += F.mse_loss(patches[i], patches[i+1])
        return loss / (len(patches) - 1)

    def optimize_universal_patch(
        self,
        frames: list,
        show_progress: bool = True
    ) -> torch.Tensor:
        """
        Optimise un patch universel sur plusieurs frames.

        Args:
            frames: Liste de tensors (C, H, W)
            show_progress: Afficher la progression

        Returns:
            Patch optimisé (C, patch_size, patch_size)
        """
        # Initialiser le patch
        patch = torch.rand(
            3, self.config.patch_size, self.config.patch_size,
            device=self.device,
            requires_grad=True
        )

        optimizer = torch.optim.Adam([patch], lr=self.config.lr)

        # Calculer les tokens de référence pour chaque frame
        print("Computing reference tokens...")
        ref_tokens = []
        for frame in frames:
            frame_tensor = frame.unsqueeze(0).to(self.device)
            with torch.no_grad():
                tokens = self._get_patch_tokens(frame_tensor)
            ref_tokens.append(tokens)

        print(f"Optimizing patch over {len(frames)} frames...")
        start_time = time.time()

        for step in range(self.config.steps):
            optimizer.zero_grad()

            total_loss = 0.0

            for i, frame in enumerate(frames):
                frame_tensor = frame.to(self.device)
                patched = self._apply_patch(frame_tensor, patch)

                tokens_adv = self._get_patch_tokens(patched.unsqueeze(0))
                loss = self._attack_loss(ref_tokens[i], tokens_adv)
                total_loss += loss

            total_loss /= len(frames)
            total_loss.backward()
            optimizer.step()

            # Clamp to valid pixel range
            patch.data.clamp_(0, 1)

            if show_progress and step % 50 == 0:
                mse = -total_loss.item()  # Inverse because we maximize
                elapsed = time.time() - start_time
                print(f"[{step:04d}/{self.config.steps}] MSE: {mse:.6f} | Time: {elapsed:.1f}s")

        print(f"Optimization complete in {time.time() - start_time:.1f}s")
        return patch.detach()

    def attack_video(
        self,
        video_path: str,
        output_path: str = None,
        patch: torch.Tensor = None,
        max_frames: int = None,
        optimize_frames: int = 50,
    ) -> dict:
        """
        Attaque une vidéo complète.

        Args:
            video_path: Chemin vers la vidéo
            output_path: Chemin de sortie (optionnel)
            patch: Patch pré-optimisé (optionnel, sinon optimise)
            max_frames: Limite de frames à traiter
            optimize_frames: Nombre de frames pour l'optimisation

        Returns:
            Dict avec résultats de l'attaque
        """
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Collecter des frames pour l'optimisation
        print("Loading frames for optimization...")
        frames = []
        frame_count = 0

        while cap.isOpened() and (max_frames is None or frame_count < max_frames):
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_tensor = self.transform(frame_rgb)
            frames.append(frame_tensor)
            frame_count += 1

        cap.release()

        # Sélectionner des frames pour l'optimisation
        if len(frames) > optimize_frames:
            indices = np.linspace(0, len(frames)-1, optimize_frames, dtype=int)
            opt_frames = [frames[i] for i in indices]
        else:
            opt_frames = frames

        # Optimiser le patch si non fourni
        if patch is None:
            patch = self.optimize_universal_patch(opt_frames)

        # Appliquer le patch à toutes les frames
        print(f"\nApplying patch to {len(frames)} frames...")
        attacked_frames = []
        mse_values = []

        for i, frame in enumerate(frames):
            frame_tensor = frame.to(self.device)

            # Référence
            with torch.no_grad():
                ref_tokens = self._get_patch_tokens(frame_tensor.unsqueeze(0))

            # Attaque
            patched = self._apply_patch(frame_tensor, patch)
            with torch.no_grad():
                adv_tokens = self._get_patch_tokens(patched.unsqueeze(0))

            mse = F.mse_loss(adv_tokens, ref_tokens).item()
            mse_values.append(mse)

            attacked_frames.append(patched.cpu())

            if i % 50 == 0:
                print(f"  Frame {i}/{len(frames)}: MSE = {mse:.6f}")

        # Sauvegarder la vidéo si demandé
        if output_path:
            self._save_video(attacked_frames, output_path, fps)

        return {
            'patch': patch.cpu(),
            'mse_values': mse_values,
            'avg_mse': np.mean(mse_values),
            'max_mse': np.max(mse_values),
            'num_frames': len(frames),
        }

    def _save_video(self, frames: list, output_path: str, fps: float):
        """Sauvegarde les frames attaquées en vidéo."""
        print(f"Saving video to {output_path}...")

        # Convertir première frame pour obtenir dimensions
        first_frame = frames[0].permute(1, 2, 0).numpy()
        first_frame = (first_frame * 255).astype(np.uint8)
        h, w = first_frame.shape[:2]

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        for frame in frames:
            frame_np = frame.permute(1, 2, 0).numpy()
            frame_np = np.clip(frame_np * 255, 0, 255).astype(np.uint8)
            frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)

        out.release()
        print(f"Video saved: {output_path}")


def main():
    """Script principal d'attaque vidéo."""
    import argparse
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description='Video adversarial attack with DINOv3')
    parser.add_argument('video_path', type=str, help='Input video path')
    parser.add_argument('--output', type=str, default=None, help='Output video path')
    parser.add_argument('--patch-size', type=int, default=32, help='Patch size')
    parser.add_argument('--steps', type=int, default=500, help='Optimization steps')
    parser.add_argument('--max-frames', type=int, default=None, help='Max frames')
    parser.add_argument('--position', type=int, nargs=2, default=[0, 0], help='Patch position (x y)')

    args = parser.parse_args()

    config = AttackConfig(
        patch_size=args.patch_size,
        patch_position=tuple(args.position),
        steps=args.steps,
    )

    attacker = VideoAttacker(attack_config=config)

    output_path = args.output or str(
        Path(args.video_path).stem + "_attacked.mp4"
    )

    results = attacker.attack_video(
        args.video_path,
        output_path=output_path,
        max_frames=args.max_frames,
    )

    # Afficher les résultats
    print(f"\n=== Attack Results ===")
    print(f"Frames processed: {results['num_frames']}")
    print(f"Average MSE: {results['avg_mse']:.6f}")
    print(f"Max MSE: {results['max_mse']:.6f}")

    # Plot MSE over time
    plt.figure(figsize=(10, 4))
    plt.plot(results['mse_values'], 'b-', alpha=0.7)
    plt.axhline(y=results['avg_mse'], color='r', linestyle='--', label=f"Avg: {results['avg_mse']:.4f}")
    plt.xlabel('Frame')
    plt.ylabel('MSE (Token Distance)')
    plt.title('Attack Effectiveness Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plot_path = str(Path(args.video_path).stem + "_attack_plot.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved: {plot_path}")

    plt.show()

    # Visualiser le patch
    plt.figure(figsize=(4, 4))
    patch_np = results['patch'].permute(1, 2, 0).numpy()
    plt.imshow(np.clip(patch_np, 0, 1))
    plt.title(f"Adversarial Patch ({args.patch_size}x{args.patch_size})")
    plt.axis('off')

    patch_path = str(Path(args.video_path).stem + "_patch.png")
    plt.savefig(patch_path, dpi=150, bbox_inches='tight')
    print(f"Patch saved: {patch_path}")

    plt.show()


if __name__ == "__main__":
    main()
