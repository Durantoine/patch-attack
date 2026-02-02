import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import KMeans


class SegmentationCoherenceAttack:
    """Adversarial patch attack targeting segmentation spatial coherence."""

    def __init__(self, model, segmenter, patch_size=50, device='cuda'):
        self.model = model
        self.segmenter = segmenter
        self.patch_size = patch_size
        self.device = device
        self.patch = nn.Parameter(torch.rand(1, 3, patch_size, patch_size, device=device))

    def apply_patch(self, image, position):
        """Apply patch to image tensor."""
        patched = image.clone()
        x, y = position
        patched[:, :, y:y+self.patch_size, x:x+self.patch_size] = self.patch
        return patched

    def extract_patch_features_diff(self, image):
        """Extract patch features with gradients enabled."""
        features = self.model.get_intermediate_layers(image, n=1)[0]
        patch_tokens = features[:, 1:, :]
        return patch_tokens

    def segment_differentiable(self, patch_tokens, kmeans_centers):
        """Differentiable soft segmentation using cluster centers.

        Args:
            patch_tokens: Patch features (B, N, D)
            kmeans_centers: Cluster centers from k-means (n_clusters, D)

        Returns:
            Soft assignment probabilities (N, n_clusters)
        """
        B, N, D = patch_tokens.shape
        tokens = patch_tokens[0]

        centers = torch.from_numpy(kmeans_centers).float().to(self.device)

        distances = torch.cdist(tokens, centers)
        soft_assignments = F.softmax(-distances * 10, dim=1)

        return soft_assignments

    def coherence_loss(self, patch_tokens, soft_assignments):
        """Loss to minimize intra-segment coherence.

        Args:
            patch_tokens: Patch features (B, N, D)
            soft_assignments: Soft cluster assignments (N, n_clusters)

        Returns:
            Coherence loss (lower = less coherent segments)
        """
        B, N, D = patch_tokens.shape
        tokens = patch_tokens[0]

        coherence = 0.0
        for k in range(soft_assignments.shape[1]):
            weights = soft_assignments[:, k]
            if weights.sum() < 1e-6:
                continue

            weighted_tokens = tokens * weights.unsqueeze(1)
            centroid = weighted_tokens.sum(dim=0) / (weights.sum() + 1e-8)

            distances = torch.norm(tokens - centroid.unsqueeze(0), dim=1)
            coherence += (distances * weights).sum()

        return -coherence

    def entropy_loss(self, soft_assignments, region_mask):
        """Maximize entropy of cluster assignments in a region.

        Args:
            soft_assignments: Soft cluster assignments (N, n_clusters)
            region_mask: Boolean mask for region of interest (H_p, W_p) or None for all

        Returns:
            Entropy loss (lower = higher entropy)
        """
        per_token_entropy = -(soft_assignments * torch.log(soft_assignments + 1e-8)).sum(dim=1)
        avg_entropy = per_token_entropy.mean()

        return -avg_entropy

    def boundary_loss(self, soft_assignments):
        """Maximize boundary length (create fragmented segments).

        Args:
            soft_assignments: Soft cluster assignments (N, n_clusters)

        Returns:
            Boundary loss (lower = more boundaries)
        """
        N, K = soft_assignments.shape
        H_p = W_p = int(np.round(np.sqrt(N)))

        if H_p * W_p != N:
            H_p = W_p = int(np.sqrt(N + 1))

        if H_p * W_p > N:
            pad_size = H_p * W_p - N
            padding = torch.zeros(pad_size, K, device=soft_assignments.device)
            soft_assignments_padded = torch.cat([soft_assignments, padding], dim=0)
        else:
            soft_assignments_padded = soft_assignments

        assignments_2d = soft_assignments_padded.reshape(H_p, W_p, K)

        horizontal_diff = torch.abs(assignments_2d[:, 1:, :] - assignments_2d[:, :-1, :]).sum()
        vertical_diff = torch.abs(assignments_2d[1:, :, :] - assignments_2d[:-1, :, :]).sum()

        boundary_score = horizontal_diff + vertical_diff

        return -boundary_score

    def attack(
        self,
        image,
        position=None,
        num_iterations=100,
        learning_rate=0.01,
        attack_mode='entropy',
        region_mask=None
    ):
        """Run segmentation coherence attack.

        Args:
            image: Image tensor (1, 3, H, W)
            position: (x, y) position for patch, None for center
            num_iterations: Number of optimization iterations
            learning_rate: Learning rate for Adam optimizer
            attack_mode: Attack objective
                - 'entropy': Maximize entropy in region (fragment segments)
                - 'coherence': Minimize segment coherence
                - 'boundary': Maximize boundary length
            region_mask: Region of interest for entropy attack (H_p, W_p)

        Returns:
            Dictionary with attack results
        """
        if position is None:
            h, w = image.shape[2:]
            position = ((w - self.patch_size) // 2, (h - self.patch_size) // 2)

        seg_orig, patch_tokens_orig, kmeans_centers = self.segmenter.segment(image)

        coherence_orig = self.segmenter.compute_segment_coherence(patch_tokens_orig, seg_orig)
        boundary_orig = self.segmenter.compute_boundary_length(seg_orig)

        optimizer = torch.optim.Adam([self.patch], lr=learning_rate)

        history = {
            'losses': [],
            'coherences': [],
            'boundaries': [],
            'iterations': []
        }

        for iteration in range(num_iterations):
            patched = self.apply_patch(image, position)
            patch_tokens = self.extract_patch_features_diff(patched)
            soft_assignments = self.segment_differentiable(patch_tokens, kmeans_centers)

            if attack_mode == 'entropy':
                if region_mask is None:
                    H_p = W_p = int(np.sqrt(soft_assignments.shape[0]))
                    region_mask = np.ones((H_p, W_p), dtype=bool)
                loss = self.entropy_loss(soft_assignments, region_mask)
            elif attack_mode == 'coherence':
                loss = self.coherence_loss(patch_tokens, soft_assignments)
            elif attack_mode == 'boundary':
                loss = self.boundary_loss(soft_assignments)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                self.patch.data.clamp_(-2, 2)

            if iteration % 10 == 0:
                with torch.no_grad():
                    seg_current, patch_tokens_current, _ = self.segmenter.segment(patched, kmeans_centers)
                    coherence_current = self.segmenter.compute_segment_coherence(
                        patch_tokens_current, seg_current
                    )
                    boundary_current = self.segmenter.compute_boundary_length(seg_current)

                history['iterations'].append(iteration)
                history['losses'].append(loss.item())
                history['coherences'].append(coherence_current)
                history['boundaries'].append(boundary_current)

                print(f"Iter {iteration:3d} | Loss: {loss.item():7.4f} | "
                      f"Coherence: {coherence_current:.4f} | Boundary: {boundary_current:.1f}")

        with torch.no_grad():
            patched_final = self.apply_patch(image, position)
            seg_final, patch_tokens_final, _ = self.segmenter.segment(patched_final, kmeans_centers)
            coherence_final = self.segmenter.compute_segment_coherence(
                patch_tokens_final, seg_final
            )
            boundary_final = self.segmenter.compute_boundary_length(seg_final)

            from ..segmentation.unsupervised import compute_iou
            iou = compute_iou(seg_orig, seg_final)

        return {
            'segmentation_original': seg_orig,
            'segmentation_attacked': seg_final,
            'coherence_original': coherence_orig,
            'coherence_attacked': coherence_final,
            'coherence_degradation': coherence_orig - coherence_final,
            'boundary_original': boundary_orig,
            'boundary_attacked': boundary_final,
            'boundary_increase': boundary_final - boundary_orig,
            'iou': iou,
            'patch': self.patch.detach().cpu(),
            'position': position,
            'history': history,
            'attack_success': (coherence_orig - coherence_final) > 0.1 or iou < 0.7
        }
