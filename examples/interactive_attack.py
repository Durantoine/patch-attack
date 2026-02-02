"""
Attaque adversariale interactive sur segmentation DINOv3
- Mode single-frame (comme test2.py) - très efficace
- Mode universel (sur plusieurs frames) - plus robuste
- Charge un patch pré-entraîné
- Déplace le patch avec la souris
- Visualise l'effet sur les embeddings en temps réel

Usage:
    # Mode single-frame (recommandé pour tester)
    python examples/interactive_attack.py --single-frame --size 500

    # Charger un patch entraîné
    python examples/interactive_attack.py --load-patch results/trained_patch.pt --size 500

    # Mode universel (30 frames)
    python examples/interactive_attack.py --size 500
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from torchvision import transforms
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import time
import argparse


# =========================
# Chargement modèle
# =========================

def load_model():
    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}")

    dinov3_path = Path(__file__).parent.parent / "src" / "models" / "dinov3"
    weights_path = Path(__file__).parent.parent / "src" / "models" / "weights" / "dinov3_vits16_pretrain_lvd1689m-08c60483.pth"

    model = torch.hub.load(str(dinov3_path), "dinov3_vits16", source="local", pretrained=False)

    if weights_path.exists():
        checkpoint = torch.load(str(weights_path), map_location=device, weights_only=False)
        model.load_state_dict(checkpoint, strict=True)

    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad = False

    return model, device


# =========================
# Fonctions utilitaires
# =========================

def get_patch_tokens(model, image: torch.Tensor, requires_grad=False) -> torch.Tensor:
    if requires_grad:
        features = model.get_intermediate_layers(image.unsqueeze(0), n=1)[0]
    else:
        with torch.no_grad():
            features = model.get_intermediate_layers(image.unsqueeze(0), n=1)[0]
    tokens = features[0, 1:]
    tokens = F.normalize(tokens, dim=1)
    return tokens


def apply_patch(image: torch.Tensor, patch: torch.Tensor, pos: tuple) -> torch.Tensor:
    x, y = pos
    patched = image.clone()
    ph, pw = patch.shape[1], patch.shape[2]
    x = max(0, min(x, image.shape[1] - ph))
    y = max(0, min(y, image.shape[2] - pw))
    patched[:, x:x+ph, y:y+pw] = patch
    return patched


def tokens_to_pca_rgb(tokens: torch.Tensor, pca_model=None) -> tuple:
    tokens_np = tokens.detach().cpu().numpy()
    if pca_model is None:
        pca_model = PCA(n_components=3)
        tokens_3d = pca_model.fit_transform(tokens_np)
    else:
        tokens_3d = pca_model.transform(tokens_np)
    tokens_3d = tokens_3d - tokens_3d.min()
    tokens_3d = tokens_3d / (tokens_3d.max() + 1e-8)
    tokens_rgb = (tokens_3d * 255).astype(np.uint8)
    return tokens_rgb, pca_model


def segment_tokens(tokens: torch.Tensor, n_clusters: int) -> np.ndarray:
    tokens_np = tokens.detach().cpu().numpy()
    kmeans = KMeans(n_clusters=n_clusters, n_init=3, max_iter=100, random_state=42)
    labels = kmeans.fit_predict(tokens_np)
    return labels


def create_pca_vis(colors: np.ndarray, size: int, smooth: bool = True) -> np.ndarray:
    h_patches, w_patches = 14, 14
    n_patches = h_patches * w_patches
    if len(colors) < n_patches:
        colors = np.vstack([colors, np.zeros((n_patches - len(colors), 3), dtype=np.uint8)])
    vis_map = colors[:n_patches].reshape(h_patches, w_patches, 3)
    interp = cv2.INTER_CUBIC if smooth else cv2.INTER_NEAREST
    return cv2.resize(vis_map, (size, size), interpolation=interp)


def create_cluster_vis(labels: np.ndarray, size: int, smooth: bool = True) -> np.ndarray:
    h_patches, w_patches = 14, 14
    n_patches = h_patches * w_patches
    if len(labels) < n_patches:
        labels = np.concatenate([labels, np.zeros(n_patches - len(labels), dtype=labels.dtype)])
    seg_map = labels[:n_patches].reshape(h_patches, w_patches)

    colors = np.array([
        [255, 50, 50], [50, 255, 50], [50, 50, 255], [255, 255, 50],
        [255, 50, 255], [50, 255, 255], [255, 150, 50], [150, 50, 255],
    ], dtype=np.uint8)

    seg_colored = np.zeros((h_patches, w_patches, 3), dtype=np.uint8)
    for i in range(len(colors)):
        seg_colored[seg_map == i] = colors[i]

    interp = cv2.INTER_CUBIC if smooth else cv2.INTER_NEAREST
    return cv2.resize(seg_colored, (size, size), interpolation=interp)


def create_distance_heatmap(distances: np.ndarray, size: int) -> np.ndarray:
    h_patches, w_patches = 14, 14
    n_patches = h_patches * w_patches
    if len(distances) < n_patches:
        distances = np.concatenate([distances, np.zeros(n_patches - len(distances))])
    dist_map = distances[:n_patches].reshape(h_patches, w_patches)
    dist_map = dist_map / (dist_map.max() + 1e-8)
    dist_resized = cv2.resize(dist_map, (size, size), interpolation=cv2.INTER_CUBIC)
    return cv2.applyColorMap((dist_resized * 255).astype(np.uint8), cv2.COLORMAP_HOT)


# =========================
# Optimisation du patch
# =========================

def collect_frames(cap, transform, device, n_frames=30):
    """Collecte N frames pour l'optimisation."""
    print(f"Collecting {n_frames} frames for optimization...")
    frames = []
    for i in range(n_frames * 2):  # Skip some frames for diversity
        ret, frame = cap.read()
        if not ret:
            break
        if i % 2 == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_tensor = transform(frame_rgb).to(device)
            frames.append(img_tensor)
        if len(frames) >= n_frames:
            break
    print(f"  Collected {len(frames)} frames")
    return frames


def optimize_single_frame(model, image: torch.Tensor, device: str,
                          patch_size: int = 32, patch_pos: tuple = (0, 0),
                          steps: int = 1000, lr: float = 0.05) -> torch.Tensor:
    """
    Optimise un patch sur UNE frame (comme test2.py).
    Très efficace, MSE élevé garanti.
    """
    print(f"\nOptimizing patch (single-frame mode, like test2.py)...")
    print(f"  Patch: {patch_size}x{patch_size} at {patch_pos}")
    print(f"  Steps: {steps}, LR: {lr}")

    patch = torch.rand(3, patch_size, patch_size, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([patch], lr=lr)

    # Tokens de référence
    with torch.no_grad():
        tokens_ref = get_patch_tokens(model, image)

    best_mse = 0
    best_patch = patch.clone()

    for step in range(steps):
        optimizer.zero_grad()

        patched_img = apply_patch(image, patch, patch_pos)
        features = model.get_intermediate_layers(patched_img.unsqueeze(0), n=1)[0]
        tokens_adv = F.normalize(features[0, 1:], dim=1)

        loss = -F.mse_loss(tokens_adv, tokens_ref)
        loss.backward()
        optimizer.step()
        patch.data.clamp_(0, 1)

        mse = -loss.item()
        if mse > best_mse:
            best_mse = mse
            best_patch = patch.detach().clone()

        if step % 100 == 0:
            cosine = F.cosine_similarity(tokens_adv.detach(), tokens_ref, dim=1).mean().item()
            print(f"  [{step:4d}/{steps}] MSE: {mse:.6f} | Cosine: {cosine:.4f}")

    print(f"  Done! Best MSE: {best_mse:.6f}")
    return best_patch


def optimize_universal_patch(model, frames: list, device: str,
                             patch_size: int = 24, steps: int = 600, lr: float = 0.05) -> torch.Tensor:
    """Optimise un patch universel sur plusieurs frames."""
    print(f"\nOptimizing universal patch on {len(frames)} frames...")
    print(f"  Patch: {patch_size}x{patch_size}")
    print(f"  Steps: {steps}, LR: {lr}")

    patch = torch.rand(3, patch_size, patch_size, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([patch], lr=lr)

    # Tokens de référence pour chaque frame
    ref_tokens = []
    for frame in frames:
        with torch.no_grad():
            tokens = get_patch_tokens(model, frame)
        ref_tokens.append(tokens)

    # Position centrale
    pos = (112 - patch_size // 2, 112 - patch_size // 2)

    best_mse = 0
    best_patch = patch.clone()

    for step in range(steps):
        optimizer.zero_grad()
        total_loss = 0.0

        # Utiliser TOUTES les frames (pas de sampling)
        for idx in range(len(frames)):
            patched_img = apply_patch(frames[idx], patch, pos)
            features = model.get_intermediate_layers(patched_img.unsqueeze(0), n=1)[0]
            tokens_adv = F.normalize(features[0, 1:], dim=1)
            loss = -F.mse_loss(tokens_adv, ref_tokens[idx])
            total_loss += loss

        total_loss /= len(frames)
        total_loss.backward()
        optimizer.step()
        patch.data.clamp_(0, 1)

        mse = -total_loss.item()
        if mse > best_mse:
            best_mse = mse
            best_patch = patch.detach().clone()

        if step % 100 == 0:
            print(f"  [{step:4d}/{steps}] MSE: {mse:.6f}")

    print(f"  Done! Best MSE: {best_mse:.6f}")
    return best_patch


def save_patch(patch: torch.Tensor, path: str):
    """Sauvegarde le patch."""
    torch.save(patch.cpu(), path)
    print(f"Patch saved to {path}")


def load_patch(path: str, device: str) -> torch.Tensor:
    """Charge un patch pré-entraîné."""
    patch = torch.load(path, map_location=device)
    print(f"Patch loaded from {path}")
    return patch


# =========================
# Application interactive
# =========================

class InteractiveAttack:
    def __init__(self, args):
        self.args = args
        self.model, self.device = load_model()
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])

        self.patch = None
        self.patch_pos = [92, 92]
        self.dragging = False
        self.pca_model = None

        # Options d'affichage
        self.mode = args.mode  # 'pca', 'segment', 'both', 'distance'
        self.n_clusters = args.clusters
        self.smooth = True
        self.show_patch = True
        self.size = args.size

    def mouse_callback(self, event, x, y, flags, param):
        size = self.size
        # Zone cliquable = première colonne d'images
        if x < size:
            img_x = int(x * 224 / size)
            img_y = int(y * 224 / size)

            if event == cv2.EVENT_LBUTTONDOWN:
                self.dragging = True
                ps = self.args.patch_size
                self.patch_pos = [img_y - ps // 2, img_x - ps // 2]

            elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
                ps = self.args.patch_size
                self.patch_pos = [img_y - ps // 2, img_x - ps // 2]
                self.patch_pos[0] = max(0, min(self.patch_pos[0], 224 - ps))
                self.patch_pos[1] = max(0, min(self.patch_pos[1], 224 - ps))

            elif event == cv2.EVENT_LBUTTONUP:
                self.dragging = False

    def run(self):
        cap = cv2.VideoCapture(self.args.camera)
        if not cap.isOpened():
            print("Cannot open camera")
            return

        # Améliorer la résolution de la caméra (Full HD)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)  # Activer l'autofocus si disponible

        actual_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print(f"Camera resolution: {int(actual_w)}x{int(actual_h)}")

        cv2.namedWindow('DINOv3 Attack')
        cv2.setMouseCallback('DINOv3 Attack', self.mouse_callback)

        # Charger ou optimiser le patch
        if self.args.load_patch and Path(self.args.load_patch).exists():
            self.patch = load_patch(self.args.load_patch, self.device)
        elif self.args.single_frame:
            # Mode single-frame (comme test2.py) - très efficace
            print("Warming up camera (30 frames)...")
            for _ in range(30):  # Warmup - laisser la caméra s'ajuster
                cap.read()

            print("Capturing frame for optimization...")
            ret, frame = cap.read()
            if ret:
                # Afficher la frame capturée
                cv2.imshow('Captured Frame', frame)
                cv2.waitKey(500)  # Montrer 0.5s

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img_tensor = self.transform(frame_rgb).to(self.device)

                # Position centrale pour le patch
                center = 112 - self.args.patch_size // 2
                patch_pos = (center, center)

                self.patch = optimize_single_frame(
                    self.model, img_tensor, self.device,
                    patch_size=self.args.patch_size,
                    patch_pos=patch_pos,
                    steps=self.args.steps,
                    lr=self.args.lr
                )
                self.patch_pos = [center, center]  # Position initiale = centre

                cv2.destroyWindow('Captured Frame')
            save_path = Path(__file__).parent.parent / "results" / "single_frame_patch.pt"
            save_path.parent.mkdir(exist_ok=True)
            save_patch(self.patch, str(save_path))
        else:
            # Mode universel
            frames = collect_frames(cap, self.transform, self.device, n_frames=self.args.n_frames)
            self.patch = optimize_universal_patch(
                self.model, frames, self.device,
                self.args.patch_size, self.args.steps, self.args.lr
            )
            save_path = Path(__file__).parent.parent / "results" / "universal_patch.pt"
            save_path.parent.mkdir(exist_ok=True)
            save_patch(self.patch, str(save_path))

        print("\n=== Controls ===")
        print("  Drag mouse: Move patch")
        print("  m: Change mode (pca/segment/both/distance)")
        print("  s: Toggle smooth")
        print("  p: Toggle patch on/off")
        print("  o: Re-optimize patch")
        print("  +/-: Change clusters")
        print("  r: Reset position")
        print("  w: Save current patch")
        print("  q: Quit")
        print()

        frame_count = 0
        fps = 0
        fps_start = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Prétraitement - garder la frame originale HD
            h, w = frame.shape[:2]
            # Crop carré au centre (comme le modèle le fait)
            if w > h:
                start_x = (w - h) // 2
                frame_square = frame[:, start_x:start_x+h]
            else:
                start_y = (h - w) // 2
                frame_square = frame[start_y:start_y+w, :]

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_tensor = self.transform(frame_rgb).to(self.device)

            # Tokens de référence
            tokens_ref = get_patch_tokens(self.model, img_tensor)

            # Appliquer le patch
            if self.show_patch and self.patch is not None:
                patched_tensor = apply_patch(img_tensor, self.patch, tuple(self.patch_pos))
                tokens_adv = get_patch_tokens(self.model, patched_tensor)
            else:
                patched_tensor = img_tensor
                tokens_adv = tokens_ref

            # Distances
            distances = torch.norm(tokens_adv - tokens_ref, dim=1).detach().cpu().numpy()
            avg_dist = distances.mean()

            # === Créer les visualisations ===
            size = self.size

            # Afficher la frame originale HD (pas le tensor 224x224)
            img_display = cv2.resize(frame_square, (size, size), interpolation=cv2.INTER_LINEAR)

            # Dessiner le patch sur l'image HD
            if self.show_patch and self.patch is not None:
                # Convertir position patch (224x224) vers display (size x size)
                # Le transform fait: Resize(256) puis CenterCrop(224)
                # Donc le patch est relatif au crop 224x224 centré
                scale = size / 224
                px, py = int(self.patch_pos[1] * scale), int(self.patch_pos[0] * scale)
                ps = int(self.args.patch_size * scale)

                # Dessiner le patch réel sur l'image
                patch_np = self.patch.detach().cpu().permute(1, 2, 0).numpy()
                patch_np = (np.clip(patch_np, 0, 1) * 255).astype(np.uint8)
                patch_np = cv2.cvtColor(patch_np, cv2.COLOR_RGB2BGR)
                patch_resized = cv2.resize(patch_np, (ps, ps), interpolation=cv2.INTER_NEAREST)

                # Appliquer le patch sur l'affichage
                y1, y2 = max(0, py), min(size, py + ps)
                x1, x2 = max(0, px), min(size, px + ps)
                py_off = max(0, -py)
                px_off = max(0, -px)
                patch_h = y2 - y1
                patch_w = x2 - x1
                if patch_h > 0 and patch_w > 0:
                    img_display[y1:y2, x1:x2] = patch_resized[py_off:py_off+patch_h, px_off:px_off+patch_w]

                # Contour du patch
                cv2.rectangle(img_display, (px, py), (px + ps, py + ps), (0, 255, 0), 2)

            # Visualisations selon le mode
            ref_colors, self.pca_model = tokens_to_pca_rgb(tokens_ref, self.pca_model)
            adv_colors, _ = tokens_to_pca_rgb(tokens_adv, self.pca_model)

            if self.mode == 'pca':
                ref_vis = create_pca_vis(ref_colors, size, self.smooth)
                adv_vis = create_pca_vis(adv_colors, size, self.smooth)
                display = np.hstack([img_display, ref_vis, adv_vis])

            elif self.mode == 'segment':
                labels_ref = segment_tokens(tokens_ref, self.n_clusters)
                labels_adv = segment_tokens(tokens_adv, self.n_clusters)
                ref_vis = create_cluster_vis(labels_ref, size, self.smooth)
                adv_vis = create_cluster_vis(labels_adv, size, self.smooth)
                display = np.hstack([img_display, ref_vis, adv_vis])

            elif self.mode == 'distance':
                dist_vis = create_distance_heatmap(distances, size)
                adv_vis = create_pca_vis(adv_colors, size, self.smooth)
                display = np.hstack([img_display, adv_vis, dist_vis])

            else:  # both
                labels_ref = segment_tokens(tokens_ref, self.n_clusters)
                labels_adv = segment_tokens(tokens_adv, self.n_clusters)
                seg_ref = create_cluster_vis(labels_ref, size, self.smooth)
                seg_adv = create_cluster_vis(labels_adv, size, self.smooth)
                pca_ref = create_pca_vis(ref_colors, size, self.smooth)
                pca_adv = create_pca_vis(adv_colors, size, self.smooth)

                row1 = np.hstack([img_display, seg_ref, seg_adv])
                row2 = np.hstack([create_distance_heatmap(distances, size), pca_ref, pca_adv])
                display = np.vstack([row1, row2])

            # FPS
            frame_count += 1
            if frame_count % 10 == 0:
                fps = frame_count / (time.time() - fps_start)
                frame_count = 0
                fps_start = time.time()

            # Info text
            patch_str = "ON" if self.show_patch else "OFF"
            smooth_str = "smooth" if self.smooth else "pixel"
            info = f"{self.mode} | k={self.n_clusters} | {smooth_str} | patch={patch_str} | dist={avg_dist:.3f} | {fps:.0f}fps"
            cv2.putText(display, info, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(display, "m:mode s:smooth p:patch +/-:k o:optimize q:quit",
                       (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

            cv2.imshow('DINOv3 Attack', display)

            # Contrôles
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('m'):
                modes = ['pca', 'segment', 'distance', 'both']
                self.mode = modes[(modes.index(self.mode) + 1) % len(modes)]
                print(f"Mode: {self.mode}")
            elif key == ord('s'):
                self.smooth = not self.smooth
                print(f"Smooth: {self.smooth}")
            elif key == ord('p'):
                self.show_patch = not self.show_patch
                print(f"Patch: {self.show_patch}")
            elif key in [ord('+'), ord('=')]:
                self.n_clusters = min(self.n_clusters + 1, 8)
                print(f"Clusters: {self.n_clusters}")
            elif key in [ord('-'), ord('_')]:
                self.n_clusters = max(self.n_clusters - 1, 2)
                print(f"Clusters: {self.n_clusters}")
            elif key == ord('r'):
                self.patch_pos = [92, 92]
                print("Position reset")
            elif key == ord('o'):
                # Re-optimize
                if self.args.single_frame:
                    print("Re-capturing frame...")
                    for _ in range(10):  # Mini warmup
                        cap.read()
                    ret, frame = cap.read()
                    if ret:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        img_tensor = self.transform(frame_rgb).to(self.device)
                        center = 112 - self.args.patch_size // 2
                        self.patch = optimize_single_frame(
                            self.model, img_tensor, self.device,
                            self.args.patch_size, (center, center), self.args.steps, self.args.lr
                        )
                        self.patch_pos = [center, center]
                else:
                    frames = collect_frames(cap, self.transform, self.device, n_frames=self.args.n_frames)
                    self.patch = optimize_universal_patch(
                        self.model, frames, self.device,
                        self.args.patch_size, self.args.steps, self.args.lr
                    )
            elif key == ord('w'):
                save_path = Path(__file__).parent.parent / "results" / f"patch_{int(time.time())}.pt"
                save_patch(self.patch, str(save_path))

        cap.release()
        cv2.destroyAllWindows()


# =========================
# Main
# =========================

def main():
    parser = argparse.ArgumentParser(
        description='Interactive DINOv3 Attack',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Mode single-frame (recommandé, comme test2.py)
  python examples/interactive_attack.py --single-frame

  # Charger un patch pré-entraîné
  python examples/interactive_attack.py --load-patch results/trained_patch.pt

  # Mode universel avec plus de frames
  python examples/interactive_attack.py --n-frames 50 --steps 800
        """
    )

    # Mode d'optimisation
    parser.add_argument('--single-frame', action='store_true',
                       help='Optimize on single frame (like test2.py, very effective)')
    parser.add_argument('--load-patch', type=str, default=None,
                       help='Load pre-trained patch (.pt file)')

    # Paramètres du patch
    parser.add_argument('--patch-size', type=int, default=32,
                       help='Patch size (default 32)')
    parser.add_argument('--steps', type=int, default=2000,
                       help='Optimization steps (default 2000)')
    parser.add_argument('--lr', type=float, default=0.05,
                       help='Learning rate (default 0.05)')
    parser.add_argument('--n-frames', type=int, default=20,
                       help='Frames for universal mode (default 20)')

    # Affichage
    parser.add_argument('--size', type=int, default=500,
                       help='Display size per panel (default 500)')
    parser.add_argument('--mode', choices=['pca', 'segment', 'distance', 'both'],
                       default='pca', help='Visualization mode')
    parser.add_argument('--clusters', type=int, default=4,
                       help='Number of clusters for segmentation')
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera index')

    args = parser.parse_args()

    app = InteractiveAttack(args)
    app.run()


if __name__ == "__main__":
    main()
