"""
Segmentation temps réel avec DINOv3 sur webcam
Compatible Mac M3 Max (MPS)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from torchvision import transforms
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


# =========================
# Chargement modèle
# =========================

def load_model():
    """Charge DINOv3 avec détection auto du device."""
    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    dinov3_path = Path(__file__).parent.parent / "src" / "models" / "dinov3"
    weights_path = Path(__file__).parent.parent / "src" / "models" / "weights" / "dinov3_vits16_pretrain_lvd1689m-08c60483.pth"

    model = torch.hub.load(
        str(dinov3_path),
        "dinov3_vits16",
        source="local",
        pretrained=False
    )

    if weights_path.exists():
        checkpoint = torch.load(str(weights_path), map_location=device, weights_only=False)
        model.load_state_dict(checkpoint, strict=True)
        print("Weights loaded")

    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad = False

    return model, device


# =========================
# Extraction features
# =========================

def get_patch_tokens(model, image: torch.Tensor) -> torch.Tensor:
    """Extrait les patch tokens normalisés."""
    with torch.no_grad():
        features = model.get_intermediate_layers(image.unsqueeze(0), n=1)[0]
        tokens = features[0, 1:]  # Remove CLS token
        tokens = F.normalize(tokens, dim=1)
    return tokens


# =========================
# Segmentation
# =========================

def segment_tokens(tokens: torch.Tensor, n_clusters: int = 4) -> np.ndarray:
    """
    Segmente les patch tokens en clusters.

    Args:
        tokens: Tensor (N, D) - patch tokens
        n_clusters: Nombre de segments

    Returns:
        labels: Array (N,) - label par patch
    """
    tokens_np = tokens.cpu().numpy()

    # KMeans est plus rapide que SpectralClustering pour le temps réel
    kmeans = KMeans(n_clusters=n_clusters, n_init=3, max_iter=100, random_state=42)
    labels = kmeans.fit_predict(tokens_np)

    return labels


def tokens_to_pca_rgb(tokens: torch.Tensor) -> np.ndarray:
    """
    Projette les tokens en RGB via PCA.
    Chaque patch devient une couleur basée sur sa représentation.

    Returns:
        rgb: Array (N, 3) - couleurs RGB normalisées [0, 255]
    """
    tokens_np = tokens.cpu().numpy()

    # PCA vers 3 dimensions
    pca = PCA(n_components=3)
    tokens_3d = pca.fit_transform(tokens_np)

    # Normaliser en [0, 255]
    tokens_3d -= tokens_3d.min(axis=0)
    tokens_3d /= (tokens_3d.max(axis=0) + 1e-8)
    tokens_3d = (tokens_3d * 255).astype(np.uint8)

    return tokens_3d


# =========================
# Visualisation
# =========================

def create_segmentation_overlay(
    frame: np.ndarray,
    labels: np.ndarray,
    patch_size: int = 16,
    alpha: float = 0.5
) -> np.ndarray:
    """
    Crée un overlay de segmentation sur l'image.

    Args:
        frame: Image BGR (H, W, 3)
        labels: Labels des patches (N,)
        patch_size: Taille des patches du modèle
        alpha: Transparence de l'overlay
    """
    h, w = frame.shape[:2]

    # Calculer la grille de patches (après crop 224x224)
    crop_size = 224
    h_patches = crop_size // patch_size
    w_patches = crop_size // patch_size

    # Reshape labels en grille
    n_patches = h_patches * w_patches
    if len(labels) != n_patches:
        # Padding si nécessaire
        labels = np.concatenate([labels, np.zeros(n_patches - len(labels), dtype=labels.dtype)])

    seg_map = labels.reshape(h_patches, w_patches)

    # Colormap pour les segments
    colors = [
        [255, 0, 0],     # Rouge
        [0, 255, 0],     # Vert
        [0, 0, 255],     # Bleu
        [255, 255, 0],   # Jaune
        [255, 0, 255],   # Magenta
        [0, 255, 255],   # Cyan
        [255, 128, 0],   # Orange
        [128, 0, 255],   # Violet
    ]

    # Créer l'image de segmentation
    seg_colored = np.zeros((h_patches, w_patches, 3), dtype=np.uint8)
    for i in range(len(colors)):
        mask = seg_map == i
        seg_colored[mask] = colors[i % len(colors)]

    # Redimensionner à la taille du crop (centre de l'image)
    seg_resized = cv2.resize(seg_colored, (crop_size, crop_size), interpolation=cv2.INTER_NEAREST)

    # Calculer la position du crop dans l'image originale
    start_y = (h - crop_size) // 2
    start_x = (w - crop_size) // 2

    # Créer l'overlay
    overlay = frame.copy()

    # Appliquer l'overlay dans la région du crop
    roi = overlay[start_y:start_y+crop_size, start_x:start_x+crop_size]
    blended = cv2.addWeighted(roi, 1-alpha, seg_resized, alpha, 0)
    overlay[start_y:start_y+crop_size, start_x:start_x+crop_size] = blended

    # Dessiner le rectangle du crop
    cv2.rectangle(overlay, (start_x, start_y), (start_x+crop_size, start_y+crop_size), (255, 255, 255), 2)

    return overlay


def create_pca_visualization(
    pca_colors: np.ndarray,
    output_size: int = 640,
    smooth: bool = True
) -> np.ndarray:
    """Crée une visualisation PCA des features en grand."""
    h_patches = 14  # 224 / 16
    w_patches = 14

    n_patches = h_patches * w_patches
    if len(pca_colors) < n_patches:
        pca_colors = np.vstack([pca_colors, np.zeros((n_patches - len(pca_colors), 3), dtype=np.uint8)])

    pca_map = pca_colors[:n_patches].reshape(h_patches, w_patches, 3)

    # Interpolation: CUBIC pour plus de définition, NEAREST pour pixels nets
    interp = cv2.INTER_CUBIC if smooth else cv2.INTER_NEAREST
    pca_resized = cv2.resize(pca_map, (output_size, output_size), interpolation=interp)

    return pca_resized


def create_cluster_visualization(
    labels: np.ndarray,
    output_size: int = 640,
    smooth: bool = True
) -> np.ndarray:
    """Crée une visualisation des clusters en grand."""
    h_patches = 14
    w_patches = 14
    n_patches = h_patches * w_patches

    # Padding si nécessaire
    if len(labels) < n_patches:
        labels = np.concatenate([labels, np.zeros(n_patches - len(labels), dtype=labels.dtype)])

    seg_map = labels[:n_patches].reshape(h_patches, w_patches)

    # Couleurs vives pour les clusters
    colors = np.array([
        [255, 50, 50],    # Rouge
        [50, 255, 50],    # Vert
        [50, 50, 255],    # Bleu
        [255, 255, 50],   # Jaune
        [255, 50, 255],   # Magenta
        [50, 255, 255],   # Cyan
        [255, 150, 50],   # Orange
        [150, 50, 255],   # Violet
    ], dtype=np.uint8)

    # Appliquer les couleurs
    seg_colored = np.zeros((h_patches, w_patches, 3), dtype=np.uint8)
    for i in range(len(colors)):
        mask = seg_map == i
        seg_colored[mask] = colors[i]

    # Redimensionner
    interp = cv2.INTER_CUBIC if smooth else cv2.INTER_NEAREST
    seg_resized = cv2.resize(seg_colored, (output_size, output_size), interpolation=interp)

    return seg_resized


# =========================
# Main
# =========================

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Real-time DINOv3 segmentation')
    parser.add_argument('--clusters', type=int, default=4, help='Number of segments')
    parser.add_argument('--mode', choices=['segment', 'pca', 'both'], default='pca',
                       help='Visualization mode')
    parser.add_argument('--camera', type=int, default=0, help='Camera index')
    parser.add_argument('--size', type=int, default=512, help='Output size (default 512)')
    parser.add_argument('--smooth', action='store_true', default=True, help='Smooth interpolation')
    parser.add_argument('--pixelated', action='store_true', help='Pixelated look (no smoothing)')
    args = parser.parse_args()

    # Charger le modèle
    model, device = load_model()
    patch_size = getattr(model, 'patch_size', 16)

    # Transform
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    # Ouvrir la webcam
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    print("\nControls:")
    print("  q: Quit")
    print("  +/-: Change number of clusters")
    print("  m: Toggle mode (segment/pca/both)")
    print()

    n_clusters = args.clusters
    mode = args.mode
    output_size = args.size
    smooth = not args.pixelated
    frame_count = 0
    fps_start = cv2.getTickCount()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Prétraitement
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_tensor = transform(frame_rgb).to(device)

        # Extraction des features
        tokens = get_patch_tokens(model, img_tensor)

        # Segmentation
        labels = segment_tokens(tokens, n_clusters=n_clusters)

        # Préparer la frame originale (redimensionnée en carré)
        h, w = frame.shape[:2]
        # Crop carré au centre (garde le max de l'image)
        if w > h:
            start_x = (w - h) // 2
            frame_square = frame[:, start_x:start_x+h]
        else:
            start_y = (h - w) // 2
            frame_square = frame[start_y:start_y+w, :]
        frame_resized = cv2.resize(frame_square, (output_size, output_size), interpolation=cv2.INTER_LINEAR)

        # Visualisation selon le mode
        if mode == 'segment':
            # Segmentation par clusters (colorée)
            seg_vis = create_cluster_visualization(labels, output_size, smooth)
            display = np.hstack([frame_resized, seg_vis])
        elif mode == 'pca':
            # PCA RGB
            pca_colors = tokens_to_pca_rgb(tokens)
            pca_vis = create_pca_visualization(pca_colors, output_size, smooth)
            display = np.hstack([frame_resized, pca_vis])
        else:  # both
            # Les 3: original | clusters | PCA
            seg_vis = create_cluster_visualization(labels, output_size, smooth)
            pca_colors = tokens_to_pca_rgb(tokens)
            pca_vis = create_pca_visualization(pca_colors, output_size, smooth)
            display = np.hstack([frame_resized, seg_vis, pca_vis])

        # Calculer FPS
        frame_count += 1
        if frame_count % 30 == 0:
            fps = frame_count / ((cv2.getTickCount() - fps_start) / cv2.getTickFrequency())
            frame_count = 0
            fps_start = cv2.getTickCount()
        else:
            fps = 0

        # Afficher infos
        smooth_str = "smooth" if smooth else "pixel"
        info_text = f"{mode} | k={n_clusters} | {smooth_str}"
        if fps > 0:
            info_text += f" | {fps:.0f}fps"
        cv2.putText(display, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display, "q:quit m:mode s:smooth +/-:clusters", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow('DINOv3 Segmentation', display)

        # Contrôles clavier (cliquer sur la fenêtre pour le focus!)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key in [ord('+'), ord('=')]:
            n_clusters = min(n_clusters + 1, 8)
            print(f"Clusters: {n_clusters}")
        elif key in [ord('-'), ord('_')]:
            n_clusters = max(n_clusters - 1, 2)
            print(f"Clusters: {n_clusters}")
        elif key == ord('m'):
            modes = ['segment', 'pca', 'both']
            mode = modes[(modes.index(mode) + 1) % len(modes)]
            print(f"Mode: {mode}")
        elif key == ord('s'):
            smooth = not smooth
            print(f"Smooth: {smooth}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
