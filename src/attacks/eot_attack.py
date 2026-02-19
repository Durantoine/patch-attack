"""
Expectation Over Transformation (EOT) Attack
Entraîne un patch robuste aux transformations géométriques et photométriques.
"""

from dataclasses import dataclass

import cv2
import kornia.geometry.transform as KG
import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class EOTConfig:
    """Configuration des transformations EOT."""

    # Rotation
    rotation_range: tuple[float, float] = (-30, 30)  # degrés

    # Échelle
    scale_range: tuple[float, float] = (0.7, 1.3)

    # Perspective (simule angle de vue)
    perspective_strength: float = 0.3

    # Couleur
    brightness_range: tuple[float, float] = (-0.2, 0.2)
    contrast_range: tuple[float, float] = (0.8, 1.2)
    saturation_range: tuple[float, float] = (0.8, 1.2)

    # Bruit
    noise_std: float = 0.05

    # Nombre de transformations par step
    n_transforms: int = 8


class PatchTransformer:
    """Applique des transformations différentiables au patch."""

    def __init__(self, config: EOTConfig, device: str = "cpu"):
        self.config = config
        self.device = device

    def random_rotation(self, patch: torch.Tensor) -> torch.Tensor:
        """Rotation aléatoire."""
        angle = torch.empty(1).uniform_(*self.config.rotation_range).to(self.device)
        # Rotation avec kornia (différentiable)
        center = torch.tensor([[patch.shape[2] / 2, patch.shape[1] / 2]]).to(self.device)
        scale = torch.ones(1).to(self.device)
        M = KG.get_rotation_matrix2d(center, angle, scale)
        rotated = KG.warp_affine(patch.unsqueeze(0), M, (patch.shape[1], patch.shape[2]))
        return rotated.squeeze(0)

    def random_scale(self, patch: torch.Tensor) -> torch.Tensor:
        """Échelle aléatoire."""
        scale = torch.empty(1).uniform_(*self.config.scale_range).item()
        new_size = int(patch.shape[1] * scale)
        if new_size < 8:
            new_size = 8

        # Resize
        scaled = F.interpolate(
            patch.unsqueeze(0), size=(new_size, new_size), mode="bilinear", align_corners=False
        ).squeeze(0)

        # Pad ou crop pour revenir à la taille originale
        orig_size = patch.shape[1]
        if new_size < orig_size:
            pad = (orig_size - new_size) // 2
            scaled = F.pad(scaled, (pad, pad, pad, pad), mode="constant", value=0)
        elif new_size > orig_size:
            start = (new_size - orig_size) // 2
            scaled = scaled[:, start : start + orig_size, start : start + orig_size]

        return scaled[:, :orig_size, :orig_size]

    def random_perspective(self, patch: torch.Tensor) -> torch.Tensor:
        """Transformation perspective aléatoire."""
        strength = self.config.perspective_strength
        h, w = patch.shape[1], patch.shape[2]

        # Points sources (coins)
        src = torch.tensor(
            [[[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]], dtype=torch.float32
        ).to(self.device)

        # Points destination (avec perturbation aléatoire)
        offset = (
            torch.empty(1, 4, 2)
            .uniform_(-strength * min(h, w), strength * min(h, w))
            .to(self.device)
        )
        dst = src + offset

        # Assurer que les points restent valides
        dst = dst.clamp(0, max(h, w) - 1)

        # Transformation perspective
        M = KG.get_perspective_transform(src, dst)
        warped = KG.warp_perspective(patch.unsqueeze(0), M, (h, w))

        return warped.squeeze(0)

    def random_color(self, patch: torch.Tensor) -> torch.Tensor:
        """Variations de couleur aléatoires."""
        # Brightness
        brightness = torch.empty(1).uniform_(*self.config.brightness_range).to(self.device)
        patch = patch + brightness

        # Contrast
        contrast = torch.empty(1).uniform_(*self.config.contrast_range).to(self.device)
        mean = patch.mean()
        patch = (patch - mean) * contrast + mean

        return patch.clamp(0, 1)

    def add_noise(self, patch: torch.Tensor) -> torch.Tensor:
        """Ajoute du bruit gaussien."""
        noise = torch.randn_like(patch) * self.config.noise_std
        return (patch + noise).clamp(0, 1)

    def transform(self, patch: torch.Tensor, apply_all: bool = True) -> torch.Tensor:
        """Applique une combinaison aléatoire de transformations."""
        transformed = patch.clone()

        if apply_all or torch.rand(1) > 0.5:
            transformed = self.random_rotation(transformed)

        if apply_all or torch.rand(1) > 0.5:
            transformed = self.random_scale(transformed)

        if apply_all or torch.rand(1) > 0.3:
            transformed = self.random_perspective(transformed)

        if apply_all or torch.rand(1) > 0.5:
            transformed = self.random_color(transformed)

        if apply_all or torch.rand(1) > 0.7:
            transformed = self.add_noise(transformed)

        return transformed


class EOTAttack:
    """Attaque avec Expectation Over Transformation."""

    def __init__(self, model, config: EOTConfig = None, device: str = "cpu"):
        self.model = model
        self.config = config or EOTConfig()
        self.device = device
        self.transformer = PatchTransformer(self.config, device)

    def get_tokens(self, image: torch.Tensor) -> torch.Tensor:
        """Extrait les patch tokens."""
        features = self.model.get_intermediate_layers(image.unsqueeze(0), n=1)[0]
        tokens = features[0, 1:]
        return F.normalize(tokens, dim=1)

    def apply_patch(
        self, image: torch.Tensor, patch: torch.Tensor, pos: tuple[int, int]
    ) -> torch.Tensor:
        """Applique le patch sur l'image."""
        x, y = pos
        patched = image.clone()
        ph, pw = patch.shape[1], patch.shape[2]
        x = max(0, min(x, image.shape[1] - ph))
        y = max(0, min(y, image.shape[2] - pw))

        # Masque pour le patch (ignorer les pixels noirs du padding)
        mask = (patch.sum(dim=0, keepdim=True) > 0.1).float()
        patched[:, x : x + ph, y : y + pw] = patch * mask + patched[:, x : x + ph, y : y + pw] * (
            1 - mask
        )

        return patched

    def optimize(
        self,
        images: list,
        patch_size: int = 50,
        steps: int = 500,
        lr: float = 0.1,
        positions: list = None,
    ) -> torch.Tensor:
        """
        Optimise un patch robuste avec EOT.

        Args:
            images: Liste de tensors (C, H, W)
            patch_size: Taille du patch
            steps: Nombre d'itérations
            lr: Learning rate
            positions: Liste de positions (optionnel, random si None)

        Returns:
            Patch optimisé
        """
        print(f"EOT Optimization: {len(images)} images, {self.config.n_transforms} transforms/step")

        # Initialiser le patch
        patch = torch.rand(3, patch_size, patch_size, device=self.device, requires_grad=True)
        optimizer = torch.optim.Adam([patch], lr=lr)

        # Calculer les tokens de référence
        ref_tokens = []
        for img in images:
            with torch.no_grad():
                tokens = self.get_tokens(img)
            ref_tokens.append(tokens)

        # Position par défaut: centre
        if positions is None:
            center = 112 - patch_size // 2
            positions = [(center, center)] * len(images)

        best_loss = float("inf")
        best_patch = patch.clone()

        for step in range(steps):
            optimizer.zero_grad()
            total_loss = 0.0

            # Sample quelques images
            n_samples = min(4, len(images))
            indices = np.random.choice(len(images), n_samples, replace=False)

            for idx in indices:
                img = images[idx]
                pos = positions[idx]

                # Appliquer N transformations au patch
                for _ in range(self.config.n_transforms):
                    # Transformer le patch
                    transformed_patch = self.transformer.transform(patch)

                    # Appliquer sur l'image
                    patched_img = self.apply_patch(img, transformed_patch, pos)

                    # Forward
                    tokens_adv = self.get_tokens(patched_img)

                    # Loss: maximiser la distance
                    loss = -F.mse_loss(tokens_adv, ref_tokens[idx])
                    total_loss += loss

            total_loss /= n_samples * self.config.n_transforms
            total_loss.backward()
            optimizer.step()
            patch.data.clamp_(0, 1)

            if total_loss.item() < best_loss:
                best_loss = total_loss.item()
                best_patch = patch.detach().clone()

            if step % 50 == 0:
                print(f"  Step {step:3d}/{steps}: MSE = {-total_loss.item():.6f}")

        print(f"  Done! Best MSE = {-best_loss:.6f}")
        return best_patch


def apply_3d_perspective(
    patch: np.ndarray,
    yaw: float = 0,
    pitch: float = 0,
    roll: float = 0,
    scale: float = 1.0,
    output_size: tuple[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Applique une transformation 3D au patch.

    Args:
        patch: Image du patch (H, W, 3) ou (H, W)
        yaw: Rotation autour de l'axe Y (gauche-droite) en degrés
        pitch: Rotation autour de l'axe X (haut-bas) en degrés
        roll: Rotation autour de l'axe Z (dans le plan) en degrés
        scale: Facteur d'échelle
        output_size: Taille de sortie (H, W), None = même que l'entrée

    Returns:
        (image_transformée, masque_alpha)
    """
    h, w = patch.shape[:2]
    if output_size is None:
        output_size = (h, w)

    # Convertir en radians
    yaw_rad = np.radians(yaw)
    pitch_rad = np.radians(pitch)
    roll_rad = np.radians(roll)

    # Points 3D du patch (carré unitaire centré)
    pts_3d = np.array(
        [[-0.5, -0.5, 0], [0.5, -0.5, 0], [0.5, 0.5, 0], [-0.5, 0.5, 0]], dtype=np.float32
    )

    # Matrices de rotation
    Rx = np.array(
        [
            [1, 0, 0],
            [0, np.cos(pitch_rad), -np.sin(pitch_rad)],
            [0, np.sin(pitch_rad), np.cos(pitch_rad)],
        ]
    )

    Ry = np.array(
        [[np.cos(yaw_rad), 0, np.sin(yaw_rad)], [0, 1, 0], [-np.sin(yaw_rad), 0, np.cos(yaw_rad)]]
    )

    Rz = np.array(
        [
            [np.cos(roll_rad), -np.sin(roll_rad), 0],
            [np.sin(roll_rad), np.cos(roll_rad), 0],
            [0, 0, 1],
        ]
    )

    # Rotation combinée
    R = Rz @ Ry @ Rx
    pts_rotated = (R @ pts_3d.T).T

    # Projection perspective simple
    f = 2.0  # Distance focale
    pts_2d = np.zeros((4, 2), dtype=np.float32)
    for i, pt in enumerate(pts_rotated):
        z = pt[2] + f
        pts_2d[i, 0] = (pt[0] / z) * f * scale
        pts_2d[i, 1] = (pt[1] / z) * f * scale

    # Convertir en coordonnées image
    pts_src = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)

    out_h, out_w = output_size
    pts_dst = pts_2d.copy()
    pts_dst[:, 0] = (pts_dst[:, 0] + 0.5) * out_w
    pts_dst[:, 1] = (pts_dst[:, 1] + 0.5) * out_h

    # Transformation perspective
    M = cv2.getPerspectiveTransform(pts_src, pts_dst)

    # Appliquer la transformation
    if len(patch.shape) == 2:
        patch = patch[:, :, np.newaxis]

    transformed = cv2.warpPerspective(
        patch,
        M,
        (out_w, out_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    # Créer le masque alpha
    mask = np.ones((h, w), dtype=np.float32)
    mask_transformed = cv2.warpPerspective(
        mask,
        M,
        (out_w, out_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    return transformed, mask_transformed


def apply_patch_with_perspective(
    image: np.ndarray,
    patch: np.ndarray,
    position: tuple[int, int],
    yaw: float = 0,
    pitch: float = 0,
    roll: float = 0,
    scale: float = 1.0,
) -> np.ndarray:
    """
    Applique un patch avec transformation 3D sur une image.

    Args:
        image: Image de fond (H, W, 3)
        patch: Image du patch (h, w, 3)
        position: Position (x, y) du centre du patch
        yaw, pitch, roll: Angles de rotation
        scale: Échelle

    Returns:
        Image avec le patch appliqué
    """
    # Transformer le patch
    patch_size = max(patch.shape[0], patch.shape[1])
    output_size = (int(patch_size * 1.5), int(patch_size * 1.5))  # Marge pour la rotation

    transformed, mask = apply_3d_perspective(patch, yaw, pitch, roll, scale, output_size)

    # Position du patch
    cx, cy = position
    h, w = transformed.shape[:2]
    x1 = cx - w // 2
    y1 = cy - h // 2
    x2 = x1 + w
    y2 = y1 + h

    # Clipper aux bords de l'image
    img_h, img_w = image.shape[:2]

    # Région valide dans l'image
    ix1 = max(0, x1)
    iy1 = max(0, y1)
    ix2 = min(img_w, x2)
    iy2 = min(img_h, y2)

    # Région correspondante dans le patch
    px1 = ix1 - x1
    py1 = iy1 - y1
    px2 = px1 + (ix2 - ix1)
    py2 = py1 + (iy2 - iy1)

    if ix2 <= ix1 or iy2 <= iy1:
        return image

    # Appliquer avec alpha blending
    result = image.copy()
    patch_region = transformed[py1:py2, px1:px2]
    mask_region = mask[py1:py2, px1:px2]

    if len(mask_region.shape) == 2:
        mask_region = mask_region[:, :, np.newaxis]

    result[iy1:iy2, ix1:ix2] = (
        patch_region * mask_region + result[iy1:iy2, ix1:ix2] * (1 - mask_region)
    ).astype(np.uint8)

    return result
