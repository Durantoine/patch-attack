import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DINOv3PatchAttack:
    """Adversarial patch attack for DINOv3 embeddings."""

    def __init__(self, model, patch_size=50, device='cuda'):
        """Initialize attack.

        Args:
            model: DINOv3 model
            patch_size: Size of square adversarial patch
            device: Device for computation
        """
        self.model = model
        self.patch_size = patch_size
        self.device = device
        self.patch = nn.Parameter(torch.rand(1, 3, patch_size, patch_size, device=device))

    def apply_patch(self, image, position):
        """Apply patch to image tensor.

        Args:
            image: Image tensor (1, 3, H, W)
            position: (x, y) tuple for patch position

        Returns:
            Patched image tensor
        """
        patched = image.clone()
        x, y = position
        patched[:, :, y:y+self.patch_size, x:x+self.patch_size] = self.patch
        return patched

    def attack_embedding(
        self,
        image,
        position=None,
        num_iterations=100,
        learning_rate=0.01,
        attack_type='maximize_distance'
    ):
        """Attack to modify embedding of a single image.

        Args:
            image: Image tensor (1, 3, H, W)
            position: (x, y) position for patch, None for center
            num_iterations: Number of optimization iterations
            learning_rate: Learning rate for Adam optimizer
            attack_type: Type of attack
                - 'maximize_distance': Maximize L2 distance from original embedding
                - 'maximize_norm': Make embedding norm as large as possible
                - 'minimize_norm': Make embedding norm as small as possible

        Returns:
            Dictionary with attack results
        """
        if position is None:
            h, w = image.shape[2:]
            position = ((w - self.patch_size) // 2, (h - self.patch_size) // 2)

        with torch.no_grad():
            feat_orig = self.model(image)
            norm_orig = feat_orig.norm().item()

        optimizer = torch.optim.Adam([self.patch], lr=learning_rate)

        history = {
            'losses': [],
            'distances': [],
            'norms': [],
            'cosine_similarities': [],
            'iterations': []
        }

        for iteration in range(num_iterations):
            patched = self.apply_patch(image, position)
            feat = self.model(patched)

            if attack_type == 'maximize_distance':
                distance = torch.norm(feat - feat_orig)
                loss = -distance
            elif attack_type == 'maximize_norm':
                loss = -feat.norm()
            elif attack_type == 'minimize_norm':
                loss = feat.norm()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                self.patch.data.clamp_(-2, 2)

            if iteration % 10 == 0:
                with torch.no_grad():
                    current_norm = feat.norm().item()
                    current_distance = torch.norm(feat - feat_orig).item()
                    current_cosine = F.cosine_similarity(feat, feat_orig).item()

                history['iterations'].append(iteration)
                history['losses'].append(loss.item())
                history['norms'].append(current_norm)
                history['distances'].append(current_distance)
                history['cosine_similarities'].append(current_cosine)

        with torch.no_grad():
            patched_final = self.apply_patch(image, position)
            feat_final = self.model(patched_final)
            norm_final = feat_final.norm().item()
            distance_final = torch.norm(feat_final - feat_orig).item()
            cosine_final = F.cosine_similarity(feat_orig, feat_final).item()

        return {
            'embedding_norm_original': norm_orig,
            'embedding_norm_attacked': norm_final,
            'l2_distance': distance_final,
            'cosine_similarity': cosine_final,
            'patch': self.patch.detach().cpu(),
            'history': history,
            'position': position,
            'attack_success': distance_final > 5.0 or cosine_final < 0.7
        }
