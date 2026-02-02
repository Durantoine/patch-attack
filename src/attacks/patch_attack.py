import torch
import torch.nn as nn
import numpy as np
from typing import Tuple


def iou_loss(pred_mask: torch.Tensor, target_mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Compute IoU (Intersection over Union) loss.

    Args:
        pred_mask: Predicted mask (B, H, W) or (H, W), values in [0, 1]
        target_mask: Target mask (B, H, W) or (H, W), values in [0, 1]
        eps: Small epsilon for numerical stability

    Returns:
        IoU loss (1 - IoU), lower is better
    """
    intersection = (pred_mask * target_mask).sum(dim=(-2, -1))
    union = pred_mask.sum(dim=(-2, -1)) + target_mask.sum(dim=(-2, -1)) - intersection
    iou = intersection / (union + eps)
    return 1 - iou.mean()


def dice_loss(pred_mask: torch.Tensor, target_mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Compute Dice loss.

    Args:
        pred_mask: Predicted mask (B, H, W) or (H, W), values in [0, 1]
        target_mask: Target mask (B, H, W) or (H, W), values in [0, 1]
        eps: Small epsilon for numerical stability

    Returns:
        Dice loss (1 - Dice), lower is better
    """
    intersection = (pred_mask * target_mask).sum(dim=(-2, -1))
    dice = (2 * intersection) / (pred_mask.sum(dim=(-2, -1)) + target_mask.sum(dim=(-2, -1)) + eps)
    return 1 - dice.mean()


def combined_segmentation_loss(
    pred_mask: torch.Tensor,
    target_mask: torch.Tensor,
    pred_logits: torch.Tensor = None,
    dice_weight: float = 1.0,
    bce_weight: float = 1.0,
) -> torch.Tensor:
    """Combined Dice + BCE loss for segmentation.

    Args:
        pred_mask: Predicted mask (B, H, W), values in [0, 1]
        target_mask: Target mask (B, H, W), values in [0, 1]
        pred_logits: Optional logits before sigmoid (B, H, W)
        dice_weight: Weight for Dice loss
        bce_weight: Weight for BCE loss

    Returns:
        Combined loss
    """
    loss = dice_weight * dice_loss(pred_mask, target_mask)

    if pred_logits is not None and bce_weight > 0:
        bce = nn.BCEWithLogitsLoss()(pred_logits, target_mask)
        loss = loss + bce_weight * bce

    return loss


class AdversarialPatch:
    """Adversarial patch for attacking segmentation models."""

    def __init__(
        self,
        patch_size: int = 50,
        image_size: Tuple[int, int] = (224, 224),
        device: str = "mps",
    ):
        """Initialize adversarial patch.

        Args:
            patch_size: Size of the square patch in pixels
            image_size: Size of input images (H, W)
            device: Device to run on
        """
        self.patch_size = patch_size
        self.image_size = image_size
        self.device = device

        # Initialize random patch (values in [0, 1])
        self.patch = torch.rand(3, patch_size, patch_size, device=device, requires_grad=True)

    def apply_patch(
        self,
        images: torch.Tensor,
        position: Tuple[int, int] = None,
    ) -> torch.Tensor:
        """Apply patch to images.

        Args:
            images: Batch of images (B, C, H, W)
            position: Position (x, y) to place patch. If None, random position

        Returns:
            Images with patch applied
        """
        batch_size = images.shape[0]
        patched_images = images.clone()

        for i in range(batch_size):
            if position is None:
                # Random position
                max_x = self.image_size[1] - self.patch_size
                max_y = self.image_size[0] - self.patch_size
                x = np.random.randint(0, max_x)
                y = np.random.randint(0, max_y)
            else:
                x, y = position

            # Apply patch
            patched_images[i, :, y:y+self.patch_size, x:x+self.patch_size] = self.patch

        return patched_images

    def optimize_patch(
        self,
        model,
        images: torch.Tensor,
        masks: torch.Tensor,
        num_iterations: int = 100,
        learning_rate: float = 0.01,
    ):
        """Optimize patch to fool segmentation model.

        Args:
            model: SAM predictor
            images: Batch of images (B, C, H, W)
            masks: Ground truth masks (B, H, W)
            num_iterations: Number of optimization steps
            learning_rate: Learning rate
        """
        optimizer = torch.optim.Adam([self.patch], lr=learning_rate)

        for iteration in range(num_iterations):
            optimizer.zero_grad()

            # Apply patch
            patched_images = self.apply_patch(images)

            # Get predictions (simplified - you'll need to adapt for SAM)
            # This is a placeholder for the actual SAM prediction logic
            loss = self._compute_loss(model, patched_images, masks)

            loss.backward()
            optimizer.step()

            # Clamp patch to [0, 1]
            with torch.no_grad():
                self.patch.clamp_(0, 1)

            if iteration % 10 == 0:
                print(f"Iteration {iteration}/{num_iterations}, Loss: {loss.item():.4f}")

    def _compute_loss(self, model, images, masks, attack_type="disappearance"):
        """Compute loss for patch optimization.

        Args:
            model: SAM predictor
            images: Patched images
            masks: Ground truth masks
            attack_type: Type of attack - "disappearance", "targeted", or "confusion"

        Returns:
            Loss value
        """
        # TODO: Implement actual SAM prediction logic
        # For now, return a placeholder
        # In practice, you would:
        # 1. Run SAM on patched images
        # 2. Get predicted masks
        # 3. Compute loss based on attack type

        # Example implementation (needs adaptation for SAM):
        # pred_masks = model.predict(images)
        #
        # if attack_type == "disappearance":
        #     # Minimize IoU to make object disappear
        #     loss = -iou_loss(pred_masks, masks)
        # elif attack_type == "targeted":
        #     # Maximize IoU with fake target
        #     loss = iou_loss(pred_masks, fake_target)
        # else:  # confusion
        #     # Combined loss for general degradation
        #     loss = combined_segmentation_loss(pred_masks, masks)

        return torch.tensor(0.0, requires_grad=True, device=self.device)

    def save(self, path: str):
        """Save patch to file."""
        torch.save(self.patch, path)

    def load(self, path: str):
        """Load patch from file."""
        self.patch = torch.load(path, map_location=self.device)
        self.patch.requires_grad = True
