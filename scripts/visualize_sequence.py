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
import matplotlib.pyplot as plt
from tqdm import tqdm

from models.dinov3_loader import load_dinov3
from utils.viz import CLASS_NAMES, CITYSCAPES_COLORS, CLASS_NAMES_SHORT, OTHER_COLOR, create_legend
from utils.config import (
    DATASET, PATCH_SIZE, PATCH_POS, VIZ_SEQ_SIZE, CLUSTERS, FPS, REFRESH,
    CLASSIFIER, SOURCE_CLASS, TARGET_CLASS, FOCUS_CLASSES, PATCH_PERSPECTIVE_MIN_SCALE,
)


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


def resize_patch(patch: torch.Tensor, size: int) -> torch.Tensor:
    """Resize patch tensor to given size."""
    return F.interpolate(
        patch.unsqueeze(0), size=(size, size), mode='bilinear', align_corners=False
    ).squeeze(0)


def compute_perspective_size(x: int, img_size: int, max_size: int, min_scale: float) -> int:
    """Effective patch size based on vertical position: smaller higher up (farther from camera)."""
    t = min(1.0, (x + max_size / 2) / img_size)
    scale = min_scale + (1.0 - min_scale) * t
    return max(1, int(max_size * scale))


def get_distance_configs(img_size: int, patch_size: int, min_scale: float) -> list[dict]:
    """Three roadside patch configs at far / medium / near distances."""
    y_road = int(img_size * 0.55)  # right-centre of road
    configs = []
    for name, x_ratio in [("Loin", 0.15), ("Moyen", 0.45), ("Proche", 0.72)]:
        x = int(img_size * x_ratio)
        eff = compute_perspective_size(x, img_size, patch_size, min_scale)
        y = min(y_road, img_size - eff - 1)
        configs.append({"name": name, "x": x, "y": y, "size": eff})
    return configs


def create_visualization(tokens, size=224, smooth=True, pca_model=None, global_min=None, global_max=None, grid=14):
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

    n_patches = grid * grid

    if len(colors) < n_patches:
        colors = np.vstack([colors, np.zeros((n_patches - len(colors), 3), dtype=np.uint8)])

    vis = colors[:n_patches].reshape(grid, grid, 3)
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


def create_segment_vis(labels, size=224, smooth=False, grid=14):
    """Create segmentation visualization."""
    n_patches = grid * grid

    if len(labels) < n_patches:
        labels = np.concatenate([labels, np.zeros(n_patches - len(labels), dtype=labels.dtype)])

    seg_map = labels[:n_patches].reshape(grid, grid)

    n_labels = int(labels.max()) + 1
    colors = generate_colors(n_labels)

    seg_colored = np.zeros((grid, grid, 3), dtype=np.uint8)
    for i in range(n_labels):
        mask = seg_map == i
        seg_colored[mask] = colors[i]

    interp = cv2.INTER_CUBIC if smooth else cv2.INTER_NEAREST
    seg_colored = cv2.resize(seg_colored, (size, size), interpolation=interp)
    return seg_colored, seg_map


def create_segment_diff(labels_ref, labels_adv, size=224, smooth=False, grid=14):
    """Create segmentation difference visualization.

    Direct comparison without label permutation (assumes consistent K-means model).
    """
    n_patches = grid * grid

    # Pad labels if needed
    if len(labels_ref) < n_patches:
        labels_ref = np.concatenate([labels_ref, np.zeros(n_patches - len(labels_ref), dtype=labels_ref.dtype)])
    if len(labels_adv) < n_patches:
        labels_adv = np.concatenate([labels_adv, np.zeros(n_patches - len(labels_adv), dtype=labels_adv.dtype)])

    labels_ref = labels_ref[:n_patches]
    labels_adv = labels_adv[:n_patches]

    # Direct comparison (no permutation needed with global K-means)
    diff = (labels_ref != labels_adv).astype(np.uint8)
    diff_map = diff.reshape(grid, grid)

    # Create colored diff (green=same, red=changed)
    diff_colored = np.zeros((grid, grid, 3), dtype=np.uint8)
    diff_colored[diff_map == 0] = [50, 150, 50]   # Green = unchanged
    diff_colored[diff_map == 1] = [255, 50, 50]   # Red = changed

    interp = cv2.INTER_CUBIC if smooth else cv2.INTER_NEAREST
    diff_colored = cv2.resize(diff_colored, (size, size), interpolation=interp)

    changed_pct = diff.sum() / len(diff) * 100
    match_pct = 100 - changed_pct
    return diff_colored, changed_pct, match_pct / 100


def create_classifier_vis(tokens, clf, size=224, smooth=False, grid=14, focus_classes=None):
    """Create semantic segmentation from classifier predictions.

    If focus_classes is set, only those classes are colored; the rest is gray.
    """
    with torch.no_grad():
        preds = clf(tokens).argmax(-1).cpu().numpy()
    n = grid * grid
    if len(preds) < n:
        preds = np.concatenate([preds, np.full(n - len(preds), preds[-1])])
    seg = preds[:n].reshape(grid, grid)
    colored = np.zeros((grid, grid, 3), dtype=np.uint8)
    if focus_classes is not None:
        colored[:] = OTHER_COLOR
        for i in focus_classes:
            colored[seg == i] = CITYSCAPES_COLORS[i]
    else:
        for i in range(19):
            colored[seg == i] = CITYSCAPES_COLORS[i]
    interp = cv2.INTER_CUBIC if smooth else cv2.INTER_NEAREST
    vis = cv2.resize(colored, (size, size), interpolation=interp)
    return vis, preds[:n]


def create_classifier_diff(preds_ref, preds_adv, source_class, target_class, size=224, smooth=False, grid=14):
    """Show where source_class tokens changed to target_class."""
    n = grid * grid
    if len(preds_ref) < n:
        preds_ref = np.concatenate([preds_ref, np.full(n - len(preds_ref), preds_ref[-1])])
    if len(preds_adv) < n:
        preds_adv = np.concatenate([preds_adv, np.full(n - len(preds_adv), preds_adv[-1])])

    diff = np.zeros((grid, grid, 3), dtype=np.uint8)
    ref_map = preds_ref[:n].reshape(grid, grid)
    adv_map = preds_adv[:n].reshape(grid, grid)

    diff[:] = [50, 50, 50]  # Gray = not source class
    source_mask = ref_map == source_class
    diff[source_mask] = [50, 200, 50]  # Green = source class unchanged
    if target_class == -1:
        fooled = source_mask & (adv_map != source_class)
    else:
        fooled = source_mask & (adv_map == target_class)
    diff[fooled] = [255, 50, 50]  # Red = successfully fooled

    interp = cv2.INTER_CUBIC if smooth else cv2.INTER_NEAREST
    diff = cv2.resize(diff, (size, size), interpolation=interp)

    n_source = source_mask.sum()
    n_fooled = fooled.sum()
    fr = n_fooled / n_source * 100 if n_source > 0 else 0
    return diff, fr, n_source


def main():
    parser = argparse.ArgumentParser(description='Visualize patch attack on image sequence')
    parser.add_argument('--dataset', type=str, default=DATASET)
    parser.add_argument('--patch', type=str, required=True)
    parser.add_argument('--patch-size', type=int, default=PATCH_SIZE)
    parser.add_argument('--patch-pos', type=int, nargs=2, default=PATCH_POS)
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--size', type=int, default=VIZ_SEQ_SIZE)
    parser.add_argument('--mode', choices=['pca', 'segment', 'both', 'trajectory', 'all', 'classifier', 'multi'], default='all')
    parser.add_argument('--perspective-min-scale', type=float, default=PATCH_PERSPECTIVE_MIN_SCALE,
                        help='Min patch scale at top of image for multi mode')
    parser.add_argument('--clusters', type=int, default=CLUSTERS)
    parser.add_argument('--fps', type=int, default=FPS)
    parser.add_argument('--smooth', action='store_true')
    parser.add_argument('--refresh', type=int, default=REFRESH)
    parser.add_argument('--classifier', type=str, default=CLASSIFIER)
    parser.add_argument('--source-class', type=int, default=SOURCE_CLASS)
    parser.add_argument('--target-class', type=int, default=TARGET_CLASS)
    parser.add_argument('--focus-classes', type=int, nargs='+', default=FOCUS_CLASSES,
                       help='Only color these classes, gray for rest. Use -1 for all.')
    args = parser.parse_args()

    focus_classes = None if args.focus_classes == [-1] else args.focus_classes

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

    # Load classifier if needed
    clf = None
    img_size = 224
    if args.classifier:
        import torch.nn as nn
        clf_data = torch.load(args.classifier, map_location=device, weights_only=False)
        clf = nn.Linear(384, 19).to(device)
        clf.load_state_dict(clf_data['state_dict'])
        clf.eval()
        img_size = clf_data.get('img_size', 224)
    elif args.mode in ('classifier', 'multi'):
        print("Error: --classifier required for classifier/multi mode")
        return
    grid = img_size // 16

    # Multi-distance setup
    dist_configs = None
    fooling_history = None
    disappeared_frames = None
    if args.mode == 'multi':
        dist_configs = get_distance_configs(img_size, args.patch_size, args.perspective_min_scale)
        fooling_history = [[] for _ in dist_configs]
        disappeared_frames = [[] for _ in dist_configs]
        print("Multi-distance configs:")
        for d in dist_configs:
            print(f"  {d['name']:8s}: row={d['x']}, col={d['y']}, size={d['size']}px")
    print(f"Image size: {img_size}x{img_size} -> {grid}x{grid} tokens")

    # Transform
    transform = transforms.Compose([
        transforms.Resize(img_size + 32),
        transforms.CenterCrop(img_size),
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
    elif args.mode in ('classifier', 'multi'):
        width = args.size * 4 + args.size // 2  # ref + 3 dist + legend
        height = args.size
    elif args.mode in ('both', 'segment'):
        width = args.size * 4  # 4 panels
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

    cv2.namedWindow('Patch Attack Visualization', cv2.WINDOW_NORMAL)
    cv2.moveWindow('Patch Attack Visualization', 100, 100)

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
        scale = size / img_size
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
                pca_model=global_pca_vis, global_min=global_pca_min, global_max=global_pca_max, grid=grid
            )
            adv_pca, _, _, _ = create_visualization(
                tokens_adv, size, smooth=args.smooth,
                pca_model=global_pca_vis, global_min=global_pca_min, global_max=global_pca_max, grid=grid
            )
            ref_pca = cv2.cvtColor(ref_pca, cv2.COLOR_RGB2BGR)
            adv_pca = cv2.cvtColor(adv_pca, cv2.COLOR_RGB2BGR)

        if args.mode == 'segment' or args.mode == 'all':
            # Use global kmeans for consistent colors across frames
            ref_labels, global_kmeans = segment_tokens(tokens_ref, args.clusters, global_kmeans)
            adv_labels, _ = segment_tokens(tokens_adv, args.clusters, global_kmeans)
            ref_seg, _ = create_segment_vis(ref_labels, size, smooth=args.smooth, grid=grid)
            adv_seg, _ = create_segment_vis(adv_labels, size, smooth=args.smooth, grid=grid)
            seg_diff, changed_pct, iou = create_segment_diff(ref_labels, adv_labels, size, smooth=args.smooth, grid=grid)
            ref_seg = cv2.cvtColor(ref_seg, cv2.COLOR_RGB2BGR)
            adv_seg = cv2.cvtColor(adv_seg, cv2.COLOR_RGB2BGR)
            seg_diff = cv2.cvtColor(seg_diff, cv2.COLOR_RGB2BGR)

        if args.mode == 'trajectory' or args.mode == 'all':
            trajectory_vis, global_pca, avg_disp, max_disp = create_trajectory_vis(
                tokens_ref, tokens_adv, size, pca_model=global_pca
            )

        if args.mode == 'classifier':
            ref_clf_vis, ref_clf_preds = create_classifier_vis(tokens_ref, clf, size, smooth=args.smooth, grid=grid, focus_classes=focus_classes)
            adv_clf_vis, adv_clf_preds = create_classifier_vis(tokens_adv, clf, size, smooth=args.smooth, grid=grid, focus_classes=focus_classes)
            clf_diff, fooling_rate, n_source = create_classifier_diff(
                ref_clf_preds, adv_clf_preds, args.source_class, args.target_class, size, smooth=args.smooth, grid=grid)
            present_classes = np.unique(np.concatenate([ref_clf_preds, adv_clf_preds]))
            clf_legend = create_legend(size, present_classes, focus_classes=focus_classes)
            ref_clf_vis = cv2.cvtColor(ref_clf_vis, cv2.COLOR_RGB2BGR)
            adv_clf_vis = cv2.cvtColor(adv_clf_vis, cv2.COLOR_RGB2BGR)
            clf_diff = cv2.cvtColor(clf_diff, cv2.COLOR_RGB2BGR)

        # Distance heatmap
        n_patches = grid * grid
        if len(distances) < n_patches:
            distances = np.concatenate([distances, np.zeros(n_patches - len(distances))])
        dist_map = distances[:n_patches].reshape(grid, grid)
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

        elif args.mode == 'classifier':
            src_n = CLASS_NAMES[args.source_class]
            tgt_n = "any" if args.target_class == -1 else CLASS_NAMES[args.target_class]
            frame = np.hstack([img_display, ref_clf_vis, adv_clf_vis, clf_diff, clf_legend])
            labels = ["Image+Patch", "Seg Original", "Seg Attacked", f"{src_n}→{tgt_n}"]
            for i, label in enumerate(labels):
                cv2.putText(frame, label, (i * size + 5, height - 8),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            cv2.putText(frame, f"Fooling: {fooling_rate:.1f}% ({n_source} tokens)", (10, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, img_path.name, (10, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        elif args.mode == 'multi':
            ref_vis_m, ref_preds_m = create_classifier_vis(
                tokens_ref, clf, size, smooth=args.smooth, grid=grid, focus_classes=focus_classes
            )
            ref_bgr = cv2.cvtColor(ref_vis_m, cv2.COLOR_RGB2BGR)
            n_source = int((ref_preds_m == args.source_class).sum())

            panels = [ref_bgr]
            for di, dcfg in enumerate(dist_configs):
                patch_d = resize_patch(patch, dcfg["size"])
                patched_d = apply_patch(img_tensor, patch_d, (dcfg["x"], dcfg["y"]))
                tokens_d = get_patch_tokens(model, patched_d)
                adv_vis_d, adv_preds_d = create_classifier_vis(
                    tokens_d, clf, size, smooth=args.smooth, grid=grid, focus_classes=focus_classes
                )
                adv_bgr = cv2.cvtColor(adv_vis_d, cv2.COLOR_RGB2BGR)

                n_adv = int((adv_preds_d == args.source_class).sum())
                fr_d = max(0.0, min(1.0, (n_source - n_adv) / n_source if n_source > 0 else 0.0))
                gone = (n_adv == 0 and n_source > 0)

                fooling_history[di].append(fr_d)
                if gone:
                    disappeared_frames[di].append(len(fooling_history[di]) - 1)

                if gone:
                    cv2.rectangle(adv_bgr, (0, 0), (size - 1, size - 1), (0, 0, 255), 4)
                    cv2.putText(adv_bgr, "DISPARU!", (size // 5, size // 2),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                else:
                    g = int(200 * fr_d)
                    r = int(200 * (1 - fr_d))
                    cv2.putText(adv_bgr, f"FR: {fr_d:.0%}", (8, size // 2),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, g, r), 2)
                cv2.putText(adv_bgr, f"{dcfg['name']} ({dcfg['size']}px)", (5, size - 8),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
                panels.append(adv_bgr)

            cv2.putText(ref_bgr, "Original", (5, size - 8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
            if n_source == 0:
                cv2.putText(ref_bgr, "Aucun humain", (5, size // 2),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150, 150, 150), 1)

            legend_m = create_legend(size, np.unique(ref_preds_m), focus_classes=focus_classes)
            frame = np.hstack(panels + [legend_m])

            src_name = CLASS_NAMES[args.source_class]
            cv2.putText(frame, f"{src_name}: {n_source} tokens | frame {len(fooling_history[0])}", (10, 22),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(frame, img_path.name, (10, 42),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

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

    # Summary plot for multi-distance mode
    if args.mode == 'multi' and fooling_history and len(fooling_history[0]) > 0:
        n_frames = len(fooling_history[0])
        frames_x = np.arange(n_frames)
        dist_colors_plot = ['#2196F3', '#FF9800', '#F44336']  # blue / orange / red

        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(14, 7), sharex=True,
            gridspec_kw={'height_ratios': [3, 1]}
        )

        for di, dcfg in enumerate(dist_configs):
            ax1.plot(frames_x, fooling_history[di], color=dist_colors_plot[di],
                    linewidth=2, label=f"{dcfg['name']} ({dcfg['size']}px)")
            if disappeared_frames[di]:
                ax1.scatter(disappeared_frames[di],
                           [fooling_history[di][f] for f in disappeared_frames[di]],
                           color=dist_colors_plot[di], marker='v', s=100, zorder=5)
        ax1.axhline(1.0, color='gray', linestyle='--', alpha=0.4, linewidth=1)
        ax1.set_ylabel("Taux de tromperie")
        ax1.set_ylim(-0.05, 1.15)
        ax1.set_title(f"Erreur de perception — '{CLASS_NAMES[args.source_class]}' — 3 distances (▼ = disparition)")
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)

        for di, dcfg in enumerate(dist_configs):
            gone_set = set(disappeared_frames[di])
            signal = np.array([1 if f in gone_set else 0 for f in range(n_frames)], dtype=float)
            pct = 100 * len(disappeared_frames[di]) // max(1, n_frames)
            ax2.fill_between(frames_x, signal, step='mid', alpha=0.65,
                            color=dist_colors_plot[di],
                            label=f"{dcfg['name']}: {len(disappeared_frames[di])} frames ({pct}%)")
        ax2.set_ylabel("Disparu")
        ax2.set_xlabel("Frame")
        ax2.set_ylim(0, 1.5)
        ax2.set_yticks([])
        ax2.legend(loc='upper right', fontsize=9)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        out_plot = Path('results/multi_distance_analysis.png')
        out_plot.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_plot, dpi=150, bbox_inches='tight')
        print(f"Summary plot saved to {out_plot}")
        plt.show()

    print("Done!")


if __name__ == "__main__":
    main()
