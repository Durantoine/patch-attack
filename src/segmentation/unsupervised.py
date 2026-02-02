import torch
import numpy as np
from sklearn.cluster import KMeans
import torch.nn.functional as F


class DINOv3Segmenter:
    """Unsupervised segmentation using DINOv3 patch tokens."""

    def __init__(self, model, n_clusters=5, patch_size=16, img_size=224):
        self.model = model
        self.n_clusters = n_clusters
        self.patch_size = patch_size
        self.img_size = img_size
        self.n_patches = img_size // patch_size

    def extract_patch_features(self, image):
        """Extract dense patch tokens from DINOv3.

        Args:
            image: Image tensor (B, 3, H, W)

        Returns:
            Patch tokens (B, N_patches, D) and spatial shape (H_p, W_p)
        """
        with torch.no_grad():
            features = self.model.get_intermediate_layers(image, n=1)[0]
            patch_tokens = features[:, 1:, :]
            B, N, D = patch_tokens.shape

            H_p = W_p = int(np.round(np.sqrt(N)))

            if H_p * W_p != N:
                H_p = W_p = int(np.sqrt(N + 1))

            if H_p * W_p > N:
                pad_size = H_p * W_p - N
                padding = torch.zeros(B, pad_size, D, device=patch_tokens.device)
                patch_tokens = torch.cat([patch_tokens, padding], dim=1)

            return patch_tokens, (H_p, W_p)

    def segment(self, image, kmeans_centers=None):
        """Perform unsupervised segmentation.

        Args:
            image: Image tensor (B, 3, H, W)
            kmeans_centers: Optional pre-computed cluster centers (n_clusters, D)

        Returns:
            Segmentation map (H_p, W_p) with cluster labels, patch tokens, kmeans
        """
        patch_tokens, (H_p, W_p) = self.extract_patch_features(image)

        tokens_np = patch_tokens[0].cpu().numpy()

        if kmeans_centers is not None:
            from sklearn.metrics.pairwise import euclidean_distances
            distances = euclidean_distances(tokens_np, kmeans_centers)
            labels = distances.argmin(axis=1)
            kmeans = None
        else:
            kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(tokens_np)
            kmeans_centers = kmeans.cluster_centers_

        segmentation_map = labels.reshape(H_p, W_p)

        return segmentation_map, patch_tokens, kmeans_centers

    def compute_segment_coherence(self, patch_tokens, segmentation_map):
        """Compute spatial coherence of segments.

        Coherence = average similarity within each segment.

        Args:
            patch_tokens: Patch features (B, N, D)
            segmentation_map: Cluster labels (H_p, W_p)

        Returns:
            Average coherence score (higher = more coherent)
        """
        B, N, D = patch_tokens.shape
        H_p, W_p = segmentation_map.shape
        tokens_2d = patch_tokens[0].reshape(H_p, W_p, D)

        coherence_scores = []
        for cluster_id in range(self.n_clusters):
            mask = segmentation_map == cluster_id
            if mask.sum() == 0:
                continue

            cluster_tokens = tokens_2d[mask]
            if len(cluster_tokens) < 2:
                continue

            pairwise_sim = F.cosine_similarity(
                cluster_tokens.unsqueeze(1),
                cluster_tokens.unsqueeze(0),
                dim=2
            )
            coherence = pairwise_sim.mean()
            coherence_scores.append(coherence.item())

        return np.mean(coherence_scores) if coherence_scores else 0.0

    def compute_boundary_length(self, segmentation_map):
        """Compute total boundary length (number of edge pixels).

        Args:
            segmentation_map: Cluster labels (H_p, W_p)

        Returns:
            Boundary length (int)
        """
        seg_tensor = torch.from_numpy(segmentation_map).float()

        horizontal_diff = (seg_tensor[:, 1:] != seg_tensor[:, :-1]).sum()
        vertical_diff = (seg_tensor[1:, :] != seg_tensor[:-1, :]).sum()

        return (horizontal_diff + vertical_diff).item()

    def visualize_segmentation(self, segmentation_map):
        """Convert segmentation map to RGB for visualization.

        Args:
            segmentation_map: Cluster labels (H_p, W_p)

        Returns:
            RGB image (H_p, W_p, 3)
        """
        import matplotlib.pyplot as plt

        colors = plt.cm.tab10(np.linspace(0, 1, self.n_clusters))[:, :3]
        seg_rgb = colors[segmentation_map]
        return seg_rgb


def compute_iou(seg1, seg2):
    """Compute best pixel accuracy between two segmentation maps with label permutation.

    Since cluster labels are arbitrary, we find the best permutation of labels
    that maximizes pixel agreement.

    Args:
        seg1: Segmentation map 1 (H, W)
        seg2: Segmentation map 2 (H, W)

    Returns:
        Best pixel accuracy (0-1)
    """
    from itertools import permutations

    seg1_flat = seg1.flatten()
    seg2_flat = seg2.flatten()

    n_labels = max(seg1.max(), seg2.max()) + 1

    if n_labels <= 10:
        from itertools import permutations
        best_acc = 0.0
        for perm in permutations(range(n_labels)):
            seg2_permuted = np.array([perm[label] if label < len(perm) else label for label in seg2_flat])
            acc = (seg1_flat == seg2_permuted).sum() / len(seg1_flat)
            best_acc = max(best_acc, acc)
        return best_acc
    else:
        from scipy.optimize import linear_sum_assignment
        confusion_matrix = np.zeros((n_labels, n_labels))
        for i in range(n_labels):
            for j in range(n_labels):
                confusion_matrix[i, j] = ((seg1_flat == i) & (seg2_flat == j)).sum()

        row_ind, col_ind = linear_sum_assignment(-confusion_matrix)
        best_match_count = confusion_matrix[row_ind, col_ind].sum()
        return best_match_count / len(seg1_flat)


def detect_boundaries(segmentation_map):
    """Detect boundary pixels in segmentation.

    Args:
        segmentation_map: Cluster labels (H, W)

    Returns:
        Binary boundary map (H, W)
    """
    seg = torch.from_numpy(segmentation_map).float()
    H, W = seg.shape

    boundaries = torch.zeros_like(seg, dtype=torch.bool)

    if H > 1:
        boundaries[:-1, :] |= seg[:-1, :] != seg[1:, :]
    if W > 1:
        boundaries[:, :-1] |= seg[:, :-1] != seg[:, 1:]

    return boundaries.numpy()
