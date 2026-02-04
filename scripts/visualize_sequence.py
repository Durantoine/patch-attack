"""
Visualise l'effet d'un patch sur une séquence d'images (comme Stuttgart).

Usage:
    python scripts/visualize_sequence.py --dataset data/stuttgart --patch results/universal_patch_final.pt

    # Sauvegarder en vidéo
    python scripts/visualize_sequence.py --dataset data/stuttgart --patch results/universal_patch_final.pt --output results/attack_video.mp4
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
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import argparse
from tqdm import tqdm

from models.dinov3_loader import load_dinov3


def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def load_model():
    """Load DINOv3 model and return model + device."""
    device = get_device()
    weights_path = Path(__file__).parent.parent / "src" / "models" / "weights" / "dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
    model = load_dinov3(checkpoint_path=str(weights_path), device=str(device))
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model, device


def get_patch_tokens(model, image):
    """Extract normalized patch tokens."""
    with torch.no_grad():
        features = model.get_intermediate_layers(image.unsqueeze(0), n=1)[0]
    tokens = features[0, 1:]  # remove CLS
    tokens = F.normalize(tokens, dim=1)
    return tokens


def apply_patch(image, patch, pos):
    """Apply patch to image."""
    x, y = pos
    patched = image.clone()
    ph, pw = patch.shape[1], patch.shape[2]
    # Clamp to image bounds
    x_end = min(x + ph, image.shape[1])
    y_end = min(y + pw, image.shape[2])
    patched[:, x:x_end, y:y_end] = patch[:, :x_end-x, :y_end-y]
    return patched


def create_visualization(tokens, size=224, smooth=True, pca_model=None, global_min=None, global_max=None):
    """Create PCA visualization from tokens.

    Returns: (vis_image, pca_model, min_vals, max_vals)
    """
    tokens_np = tokens.cpu().numpy()

    if pca_model is None:
        pca_model = PCA(n_components=3)
        pca_model.fit(tokens_np)

    tokens_3d = pca_model.transform(tokens_np)

    # Compute or use global min/max
    if global_min is None:
        global_min = tokens_3d.min(axis=0)
    if global_max is None:
        global_max = tokens_3d.max(axis=0)

    # Normalize with global range
    range_vals = global_max - global_min + 1e-8
    tokens_3d = (tokens_3d - global_min) / range_vals
    tokens_3d = np.clip(tokens_3d, 0, 1)
    colors = (tokens_3d * 255).astype(np.uint8)

    h_patches = w_patches = 14
    n_patches = h_patches * w_patches

    if len(colors) < n_patches:
        colors = np.vstack([colors, np.zeros((n_patches - len(colors), 3), dtype=np.uint8)])

    vis = colors[:n_patches].reshape(h_patches, w_patches, 3)
    interp = cv2.INTER_CUBIC if smooth else cv2.INTER_NEAREST
    vis = cv2.resize(vis, (size, size), interpolation=interp)
    return vis, pca_model, global_min, global_max


def create_trajectory_vis(tokens_ref, tokens_adv, size=400, pca_model=None):
    """Create PCA trajectory visualization showing token displacement.

    Returns an image showing:
    - Blue dots: original token positions
    - Red dots: attacked token positions
    - Arrows: displacement vectors (colored by magnitude)
    """
    ref_np = tokens_ref.cpu().numpy()
    adv_np = tokens_adv.cpu().numpy()

    # Fit PCA on combined tokens for consistent projection
    if pca_model is None:
        all_tokens = np.vstack([ref_np, adv_np])
        pca_model = PCA(n_components=2)
        pca_model.fit(all_tokens)

    ref_2d = pca_model.transform(ref_np)
    adv_2d = pca_model.transform(adv_np)

    # Compute displacement magnitudes for coloring
    displacements = np.linalg.norm(adv_2d - ref_2d, axis=1)
    max_disp = displacements.max() if displacements.max() > 0 else 1

    # Create blank image (dark background)
    img = np.zeros((size, size, 3), dtype=np.uint8)
    img[:] = (30, 30, 30)  # Dark gray background

    # Normalize coordinates to image space with margin
    margin = 40
    all_points = np.vstack([ref_2d, adv_2d])
    min_vals = all_points.min(axis=0)
    max_vals = all_points.max(axis=0)
    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1  # Avoid division by zero

    def to_pixel(pt):
        normalized = (pt - min_vals) / range_vals
        x = int(margin + normalized[0] * (size - 2 * margin))
        y = int(margin + normalized[1] * (size - 2 * margin))
        return (x, y)

    # Draw arrows (displacement vectors) - colored by magnitude
    for i in range(len(ref_2d)):
        p1 = to_pixel(ref_2d[i])
        p2 = to_pixel(adv_2d[i])

        # Color by displacement magnitude (blue->yellow->red)
        intensity = displacements[i] / max_disp
        if intensity < 0.5:
            # Blue to Yellow
            r = int(255 * intensity * 2)
            g = int(255 * intensity * 2)
            b = int(255 * (1 - intensity * 2))
        else:
            # Yellow to Red
            r = 255
            g = int(255 * (1 - (intensity - 0.5) * 2))
            b = 0

        color = (b, g, r)  # BGR for OpenCV
        cv2.arrowedLine(img, p1, p2, color, 1, tipLength=0.3)

    # Draw reference points (blue, smaller)
    for i in range(len(ref_2d)):
        pt = to_pixel(ref_2d[i])
        cv2.circle(img, pt, 3, (255, 150, 50), -1)  # Blue

    # Draw adversarial points (red, smaller)
    for i in range(len(adv_2d)):
        pt = to_pixel(adv_2d[i])
        cv2.circle(img, pt, 3, (50, 50, 255), -1)  # Red

    # Add legend
    cv2.circle(img, (size - 80, 20), 5, (255, 150, 50), -1)
    cv2.putText(img, "Clean", (size - 70, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    cv2.circle(img, (size - 80, 40), 5, (50, 50, 255), -1)
    cv2.putText(img, "Attack", (size - 70, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    # Add axes labels
    cv2.putText(img, "PC1", (size // 2, size - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
    cv2.putText(img, "PC2", (5, size // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

    return img, pca_model, displacements.mean(), displacements.max()


def segment_tokens(tokens, n_clusters=4, kmeans_model=None):
    """Segment tokens using K-means.

    If kmeans_model is provided, use it to predict labels (for consistency across frames).
    Otherwise, fit a new model and return it along with labels.
    """
    tokens_np = tokens.cpu().numpy()

    if kmeans_model is None:
        kmeans_model = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
        labels = kmeans_model.fit_predict(tokens_np)
    else:
        labels = kmeans_model.predict(tokens_np)

    return labels, kmeans_model


def generate_colors(n):
    """Generate n distinct colors using HSV color space."""
    colors = []
    for i in range(n):
        hue = int(180 * i / n)  # OpenCV hue range 0-179
        hsv = np.uint8([[[hue, 220, 230]]])
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0, 0]
        colors.append(rgb)
    return np.array(colors, dtype=np.uint8)


def create_segment_vis(labels, size=224, smooth=False):
    """Create segmentation visualization."""
    h_patches = w_patches = 14
    n_patches = h_patches * w_patches

    if len(labels) < n_patches:
        labels = np.concatenate([labels, np.zeros(n_patches - len(labels), dtype=labels.dtype)])

    seg_map = labels[:n_patches].reshape(h_patches, w_patches)

    n_labels = int(labels.max()) + 1
    colors = generate_colors(n_labels)

    seg_colored = np.zeros((h_patches, w_patches, 3), dtype=np.uint8)
    for i in range(n_labels):
        mask = seg_map == i
        seg_colored[mask] = colors[i]

    interp = cv2.INTER_CUBIC if smooth else cv2.INTER_NEAREST
    seg_colored = cv2.resize(seg_colored, (size, size), interpolation=interp)
    return seg_colored, seg_map


def create_segment_diff(labels_ref, labels_adv, size=224, smooth=False):
    """Create segmentation difference visualization.

    Direct comparison without label permutation (assumes consistent K-means model).
    """
    h_patches = w_patches = 14
    n_patches = h_patches * w_patches

    # Pad labels if needed
    if len(labels_ref) < n_patches:
        labels_ref = np.concatenate([labels_ref, np.zeros(n_patches - len(labels_ref), dtype=labels_ref.dtype)])
    if len(labels_adv) < n_patches:
        labels_adv = np.concatenate([labels_adv, np.zeros(n_patches - len(labels_adv), dtype=labels_adv.dtype)])

    labels_ref = labels_ref[:n_patches]
    labels_adv = labels_adv[:n_patches]

    # Direct comparison (no permutation needed with global K-means)
    diff = (labels_ref != labels_adv).astype(np.uint8)
    diff_map = diff.reshape(h_patches, w_patches)

    # Create colored diff (green=same, red=changed)
    diff_colored = np.zeros((h_patches, w_patches, 3), dtype=np.uint8)
    diff_colored[diff_map == 0] = [50, 150, 50]   # Green = unchanged
    diff_colored[diff_map == 1] = [255, 50, 50]   # Red = changed

    interp = cv2.INTER_CUBIC if smooth else cv2.INTER_NEAREST
    diff_colored = cv2.resize(diff_colored, (size, size), interpolation=interp)

    changed_pct = diff.sum() / len(diff) * 100
    match_pct = 100 - changed_pct
    return diff_colored, changed_pct, match_pct / 100


def main():
    parser = argparse.ArgumentParser(description='Visualize patch attack on image sequence')
    parser.add_argument('--dataset', type=str, required=True, help='Path to image folder')
    parser.add_argument('--patch', type=str, required=True, help='Path to trained patch (.pt)')
    parser.add_argument('--patch-size', type=int, default=32, help='Patch size')
    parser.add_argument('--patch-pos', type=int, nargs=2, default=[50, 50], help='Patch position (x, y)')
    parser.add_argument('--output', type=str, default=None, help='Output video path (optional)')
    parser.add_argument('--size', type=int, default=300, help='Visualization size')
    parser.add_argument('--mode', choices=['pca', 'segment', 'both', 'trajectory', 'all'], default='all', help='Visualization mode')
    parser.add_argument('--clusters', type=int, default=4, help='Number of clusters for segmentation')
    parser.add_argument('--fps', type=int, default=10, help='Output video FPS')
    parser.add_argument('--smooth', action='store_true', help='Use smooth interpolation (default: sharp/nearest)')
    parser.add_argument('--refresh', type=int, default=50, help='Refresh models every N frames (for changing scenes)')
    args = parser.parse_args()

    # Load model
    print("Loading model...")
    model, device = load_model()

    # Load patch
    print(f"Loading patch from {args.patch}")
    patch = torch.load(args.patch, map_location=device, weights_only=True)
    patch = patch.to(device)

    # Resize patch if needed
    if patch.shape[1] != args.patch_size or patch.shape[2] != args.patch_size:
        patch = F.interpolate(patch.unsqueeze(0), size=(args.patch_size, args.patch_size),
                             mode='bilinear', align_corners=False).squeeze(0)

    print(f"Patch shape: {patch.shape}")

    # Transform
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    # Find images
    dataset_path = Path(args.dataset)
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.webp']
    image_paths = []
    for ext in extensions:
        image_paths.extend(sorted(dataset_path.glob(ext)))

    print(f"Found {len(image_paths)} images")

    if len(image_paths) == 0:
        print("No images found!")
        return

    # Setup video writer
    if args.mode == 'all':
        width = args.size * 4   # 4 columns
        height = args.size * 2  # 2 rows
    elif args.mode == 'both' or args.mode == 'segment':
        width = args.size * 4  # Original | Ref | Adv | Diff/Distance
        height = args.size
    elif args.mode == 'trajectory':
        width = args.size * 2  # Image | Trajectory plot
        height = args.size
    else:
        width = args.size * 3  # Original | Ref | Adv
        height = args.size

    # For trajectory mode, maintain a global PCA model for consistency across frames
    global_pca = None
    # For segmentation, maintain a global K-means model for consistent colors
    global_kmeans = None
    # For PCA visualization, maintain consistent model and normalization
    global_pca_vis = None
    global_pca_min = None
    global_pca_max = None
    # Counter for model refresh
    frame_count = 0

    video_writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(args.output, fourcc, args.fps, (width, height))
        print(f"Saving video to {args.output}")

    # Process images
    print("\nProcessing images...")
    patch_pos = tuple(args.patch_pos)

    for img_path in tqdm(image_paths):
        # Load image
        image = Image.open(img_path).convert('RGB')
        img_tensor = transform(image).to(device)

        # Get reference tokens (without patch)
        tokens_ref = get_patch_tokens(model, img_tensor)

        # Apply patch and get adversarial tokens
        patched_img = apply_patch(img_tensor, patch, patch_pos)
        tokens_adv = get_patch_tokens(model, patched_img)

        # Calculate distances
        distances = torch.norm(tokens_adv - tokens_ref, dim=1).cpu().numpy()
        mse = F.mse_loss(tokens_adv, tokens_ref).item()

        # Create visualizations
        size = args.size

        # Original image with patch
        img_np = patched_img.cpu().permute(1, 2, 0).numpy()
        img_np = (np.clip(img_np, 0, 1) * 255).astype(np.uint8)
        img_display = cv2.resize(img_np, (size, size))
        img_display = cv2.cvtColor(img_display, cv2.COLOR_RGB2BGR)

        # Draw patch rectangle
        scale = size / 224
        px, py = int(patch_pos[1] * scale), int(patch_pos[0] * scale)
        ps = int(args.patch_size * scale)
        cv2.rectangle(img_display, (px, py), (px + ps, py + ps), (0, 255, 0), 2)

        # Refresh models periodically to adapt to scene changes
        if args.refresh > 0 and frame_count % args.refresh == 0:
            global_pca_vis = None
            global_pca_min = None
            global_pca_max = None
            global_kmeans = None
            global_pca = None
        frame_count += 1

        if args.mode == 'pca' or args.mode == 'both' or args.mode == 'all':
            ref_pca, global_pca_vis, global_pca_min, global_pca_max = create_visualization(
                tokens_ref, size, smooth=args.smooth,
                pca_model=global_pca_vis, global_min=global_pca_min, global_max=global_pca_max
            )
            adv_pca, _, _, _ = create_visualization(
                tokens_adv, size, smooth=args.smooth,
                pca_model=global_pca_vis, global_min=global_pca_min, global_max=global_pca_max
            )
            ref_pca = cv2.cvtColor(ref_pca, cv2.COLOR_RGB2BGR)
            adv_pca = cv2.cvtColor(adv_pca, cv2.COLOR_RGB2BGR)

        if args.mode == 'segment' or args.mode == 'all':
            # Use global kmeans for consistent colors across frames
            ref_labels, global_kmeans = segment_tokens(tokens_ref, args.clusters, global_kmeans)
            adv_labels, _ = segment_tokens(tokens_adv, args.clusters, global_kmeans)
            ref_seg, _ = create_segment_vis(ref_labels, size, smooth=args.smooth)
            adv_seg, _ = create_segment_vis(adv_labels, size, smooth=args.smooth)
            seg_diff, changed_pct, iou = create_segment_diff(ref_labels, adv_labels, size, smooth=args.smooth)
            ref_seg = cv2.cvtColor(ref_seg, cv2.COLOR_RGB2BGR)
            adv_seg = cv2.cvtColor(adv_seg, cv2.COLOR_RGB2BGR)
            seg_diff = cv2.cvtColor(seg_diff, cv2.COLOR_RGB2BGR)

        if args.mode == 'trajectory' or args.mode == 'all':
            trajectory_vis, global_pca, avg_disp, max_disp = create_trajectory_vis(
                tokens_ref, tokens_adv, size, pca_model=global_pca
            )

        # Distance heatmap
        h_patches = w_patches = 14
        n_patches = h_patches * w_patches
        if len(distances) < n_patches:
            distances = np.concatenate([distances, np.zeros(n_patches - len(distances))])
        dist_map = distances[:n_patches].reshape(h_patches, w_patches)
        interp = cv2.INTER_CUBIC if args.smooth else cv2.INTER_NEAREST
        dist_map = cv2.resize(dist_map, (size, size), interpolation=interp)
        dist_map = (dist_map / dist_map.max() * 255).astype(np.uint8)
        dist_heatmap = cv2.applyColorMap(dist_map, cv2.COLORMAP_HOT)

        # Combine
        if args.mode == 'all':
            # 2x4 grid layout:
            # Row 1: Image | PCA Orig | PCA Attack | Trajectory
            # Row 2: Distance | Seg Orig | Seg Attack | Seg Diff
            row1 = np.hstack([img_display, ref_pca, adv_pca, trajectory_vis])
            row2 = np.hstack([dist_heatmap, ref_seg, adv_seg, seg_diff])
            frame = np.vstack([row1, row2])

            # Add labels on each panel
            labels_row1 = ["Image+Patch", "PCA Orig", "PCA Attack", "Trajectories"]
            labels_row2 = ["Distance", "Seg Orig", "Seg Attack", "Seg Diff"]
            for i, label in enumerate(labels_row1):
                cv2.putText(frame, label, (i * size + 5, size - 8),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
            for i, label in enumerate(labels_row2):
                cv2.putText(frame, label, (i * size + 5, 2 * size - 8),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

            # Add metrics
            cv2.putText(frame, f"MSE: {mse:.4f} | Changed: {changed_pct:.1f}%", (10, 22),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(frame, img_path.name, (10, 42),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

        elif args.mode == 'both':
            frame = np.hstack([img_display, ref_pca, adv_pca, dist_heatmap])
            labels = ["Image + Patch", "PCA Original", "PCA Attacked", "Distance"]
            for i, label in enumerate(labels):
                cv2.putText(frame, label, (i * size + 5, height - 8),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            cv2.putText(frame, f"MSE: {mse:.6f}", (10, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, img_path.name, (10, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        elif args.mode == 'pca':
            frame = np.hstack([img_display, ref_pca, adv_pca])
            labels = ["Image + Patch", "PCA Original", "PCA Attacked"]
            for i, label in enumerate(labels):
                cv2.putText(frame, label, (i * size + 5, height - 8),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            cv2.putText(frame, f"MSE: {mse:.6f}", (10, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, img_path.name, (10, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        elif args.mode == 'trajectory':
            frame = np.hstack([img_display, trajectory_vis])
            labels = ["Image + Patch", "Token Trajectories (PCA)"]
            for i, label in enumerate(labels):
                cv2.putText(frame, label, (i * size + 5, height - 8),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            cv2.putText(frame, f"Avg disp: {avg_disp:.4f} | Max: {max_disp:.4f}", (10, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, img_path.name, (10, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        else:  # segment
            frame = np.hstack([img_display, ref_seg, adv_seg, seg_diff])
            labels = ["Image + Patch", "Seg Original", "Seg Attacked", "Changed"]
            for i, label in enumerate(labels):
                cv2.putText(frame, label, (i * size + 5, height - 8),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            cv2.putText(frame, f"Changed: {changed_pct:.1f}% | IoU: {iou:.2f}", (10, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, img_path.name, (10, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        # Show or save
        if video_writer:
            video_writer.write(frame)

        cv2.imshow('Patch Attack Visualization', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):  # Pause
            cv2.waitKey(0)

    if video_writer:
        video_writer.release()
        print(f"\nVideo saved to {args.output}")

    cv2.destroyAllWindows()
    print("Done!")


if __name__ == "__main__":
    main()
