"""
Interactive EOT Attack Demo
- Entraîne un patch robuste aux transformations (EOT)
- Simule le patch avec orientation 3D contrôlable
- Visualise l'effet sur la segmentation en temps réel
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

from attacks.eot_attack import EOTAttack, EOTConfig, apply_patch_with_perspective


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
# Utilitaires
# =========================

def get_tokens(model, image: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        features = model.get_intermediate_layers(image.unsqueeze(0), n=1)[0]
    tokens = features[0, 1:]
    return F.normalize(tokens, dim=1)


def tokens_to_pca(tokens: torch.Tensor, pca_model=None):
    tokens_np = tokens.detach().cpu().numpy()
    if pca_model is None:
        pca_model = PCA(n_components=3)
        tokens_3d = pca_model.fit_transform(tokens_np)
    else:
        tokens_3d = pca_model.transform(tokens_np)
    tokens_3d = tokens_3d - tokens_3d.min()
    tokens_3d = tokens_3d / (tokens_3d.max() + 1e-8)
    return (tokens_3d * 255).astype(np.uint8), pca_model


def segment_tokens(tokens: torch.Tensor, n_clusters: int) -> np.ndarray:
    tokens_np = tokens.detach().cpu().numpy()
    kmeans = KMeans(n_clusters=n_clusters, n_init=3, max_iter=100, random_state=42)
    return kmeans.fit_predict(tokens_np)


def create_vis(data: np.ndarray, size: int, is_labels: bool = False, smooth: bool = True) -> np.ndarray:
    h_patches, w_patches = 14, 14
    n_patches = h_patches * w_patches

    if is_labels:
        if len(data) < n_patches:
            data = np.concatenate([data, np.zeros(n_patches - len(data), dtype=data.dtype)])
        seg_map = data[:n_patches].reshape(h_patches, w_patches)
        colors = np.array([
            [255, 50, 50], [50, 255, 50], [50, 50, 255], [255, 255, 50],
            [255, 50, 255], [50, 255, 255], [255, 150, 50], [150, 50, 255],
        ], dtype=np.uint8)
        vis = np.zeros((h_patches, w_patches, 3), dtype=np.uint8)
        for i in range(len(colors)):
            vis[seg_map == i] = colors[i]
    else:
        if len(data) < n_patches:
            data = np.vstack([data, np.zeros((n_patches - len(data), 3), dtype=np.uint8)])
        vis = data[:n_patches].reshape(h_patches, w_patches, 3)

    interp = cv2.INTER_CUBIC if smooth else cv2.INTER_NEAREST
    return cv2.resize(vis, (size, size), interpolation=interp)


def collect_frames(cap, transform, device, n_frames=30):
    print(f"Collecting {n_frames} frames...")
    frames = []
    for i in range(n_frames * 2):
        ret, frame = cap.read()
        if not ret:
            break
        if i % 2 == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_tensor = transform(frame_rgb).to(device)
            frames.append(img_tensor)
        if len(frames) >= n_frames:
            break
    return frames


# =========================
# Application principale
# =========================

class EOTInteractiveDemo:
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
        self.patch_np = None

        # Position et orientation 3D
        self.pos_x = 112
        self.pos_y = 112
        self.yaw = 0      # Gauche-droite
        self.pitch = 0    # Haut-bas
        self.roll = 0     # Rotation dans le plan
        self.scale = 1.0

        # Options d'affichage
        self.mode = args.mode
        self.n_clusters = args.clusters
        self.smooth = True
        self.show_patch = True
        self.pca_model = None

        # Mouse drag
        self.dragging = False

    def mouse_callback(self, event, x, y, flags, param):
        size = self.args.size
        if x < size:
            img_x = int(x * 224 / size)
            img_y = int(y * 224 / size)

            if event == cv2.EVENT_LBUTTONDOWN:
                self.dragging = True
                self.pos_x = img_x
                self.pos_y = img_y
            elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
                self.pos_x = img_x
                self.pos_y = img_y
            elif event == cv2.EVENT_LBUTTONUP:
                self.dragging = False

    def apply_patch_to_tensor(self, img_tensor: torch.Tensor) -> torch.Tensor:
        """Applique le patch avec transformation 3D sur le tensor."""
        # Convertir en numpy pour appliquer la perspective
        img_np = (img_tensor.cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)

        # Appliquer le patch avec perspective
        result = apply_patch_with_perspective(
            img_np, self.patch_np,
            (self.pos_x, self.pos_y),
            self.yaw, self.pitch, self.roll, self.scale
        )

        # Reconvertir en tensor
        result_tensor = torch.from_numpy(result).permute(2, 0, 1).float() / 255.0
        return result_tensor.to(self.device)

    def run(self):
        cap = cv2.VideoCapture(self.args.camera)
        if not cap.isOpened():
            print("Cannot open camera")
            return

        cv2.namedWindow('EOT Attack')
        cv2.setMouseCallback('EOT Attack', self.mouse_callback)

        # Charger ou entraîner le patch
        patch_path = Path(__file__).parent.parent / "results" / "eot_patch.pt"

        if self.args.load_patch and Path(self.args.load_patch).exists():
            self.patch = torch.load(self.args.load_patch, map_location=self.device)
            print(f"Loaded patch from {self.args.load_patch}")
        elif patch_path.exists() and not self.args.retrain:
            self.patch = torch.load(str(patch_path), map_location=self.device)
            print(f"Loaded existing EOT patch")
        else:
            # Entraîner avec EOT
            frames = collect_frames(cap, self.transform, self.device, self.args.n_frames)

            eot_config = EOTConfig(
                rotation_range=(-30, 30),
                scale_range=(0.7, 1.3),
                perspective_strength=0.3,
                n_transforms=self.args.n_transforms,
            )

            attack = EOTAttack(self.model, eot_config, self.device)
            self.patch = attack.optimize(
                frames,
                patch_size=self.args.patch_size,
                steps=self.args.steps,
                lr=self.args.lr,
            )

            # Sauvegarder
            patch_path.parent.mkdir(exist_ok=True)
            torch.save(self.patch, str(patch_path))
            print(f"Saved EOT patch to {patch_path}")

        # Convertir le patch en numpy pour l'affichage
        self.patch_np = (self.patch.cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        self.patch_np = cv2.cvtColor(self.patch_np, cv2.COLOR_RGB2BGR)

        print("\n=== Controls ===")
        print("  Mouse drag: Move patch position")
        print("  Arrow keys: Adjust yaw/pitch")
        print("  r/f: Roll left/right")
        print("  +/-: Scale up/down")
        print("  0: Reset orientation")
        print("  m: Change mode")
        print("  s: Toggle smooth")
        print("  p: Toggle patch")
        print("  o: Re-train patch")
        print("  w: Save patch")
        print("  q: Quit")
        print()

        frame_count = 0
        fps = 0
        fps_start = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Prétraitement
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_tensor = self.transform(frame_rgb).to(self.device)

            # Tokens de référence
            tokens_ref = get_tokens(self.model, img_tensor)

            # Appliquer le patch avec perspective
            if self.show_patch and self.patch is not None:
                patched_tensor = self.apply_patch_to_tensor(img_tensor)
                tokens_adv = get_tokens(self.model, patched_tensor)
            else:
                patched_tensor = img_tensor
                tokens_adv = tokens_ref

            # Distance
            distances = torch.norm(tokens_adv - tokens_ref, dim=1).cpu().numpy()
            avg_dist = distances.mean()

            # === Visualisations ===
            size = self.args.size

            # Image avec patch
            img_np = (patched_tensor.cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            img_display = cv2.resize(img_np, (size, size))

            # Selon le mode
            ref_colors, self.pca_model = tokens_to_pca(tokens_ref, self.pca_model)
            adv_colors, _ = tokens_to_pca(tokens_adv, self.pca_model)

            if self.mode == 'pca':
                ref_vis = create_vis(ref_colors, size, is_labels=False, smooth=self.smooth)
                adv_vis = create_vis(adv_colors, size, is_labels=False, smooth=self.smooth)
                display = np.hstack([img_display, ref_vis, adv_vis])

            elif self.mode == 'segment':
                labels_ref = segment_tokens(tokens_ref, self.n_clusters)
                labels_adv = segment_tokens(tokens_adv, self.n_clusters)
                ref_vis = create_vis(labels_ref, size, is_labels=True, smooth=self.smooth)
                adv_vis = create_vis(labels_adv, size, is_labels=True, smooth=self.smooth)
                display = np.hstack([img_display, ref_vis, adv_vis])

            else:  # both
                labels_ref = segment_tokens(tokens_ref, self.n_clusters)
                labels_adv = segment_tokens(tokens_adv, self.n_clusters)
                seg_ref = create_vis(labels_ref, size, is_labels=True, smooth=self.smooth)
                seg_adv = create_vis(labels_adv, size, is_labels=True, smooth=self.smooth)
                pca_ref = create_vis(ref_colors, size, is_labels=False, smooth=self.smooth)
                pca_adv = create_vis(adv_colors, size, is_labels=False, smooth=self.smooth)

                row1 = np.hstack([img_display, seg_ref, seg_adv])
                row2 = np.hstack([
                    cv2.resize(self.patch_np, (size, size)),
                    pca_ref,
                    pca_adv
                ])
                display = np.vstack([row1, row2])

            # FPS
            frame_count += 1
            if frame_count % 10 == 0:
                fps = frame_count / (time.time() - fps_start)
                frame_count = 0
                fps_start = time.time()

            # Info
            orient_info = f"yaw={self.yaw:.0f} pitch={self.pitch:.0f} roll={self.roll:.0f} scale={self.scale:.1f}"
            info = f"{self.mode} | {orient_info} | dist={avg_dist:.3f} | {fps:.0f}fps"
            cv2.putText(display, info, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
            cv2.putText(display, "Arrows:yaw/pitch r/f:roll +/-:scale 0:reset m:mode p:patch q:quit",
                       (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

            cv2.imshow('EOT Attack', display)

            # Contrôles
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == 81 or key == 2:  # Left arrow
                self.yaw -= 5
            elif key == 83 or key == 3:  # Right arrow
                self.yaw += 5
            elif key == 82 or key == 0:  # Up arrow
                self.pitch -= 5
            elif key == 84 or key == 1:  # Down arrow
                self.pitch += 5
            elif key == ord('r'):
                self.roll -= 5
            elif key == ord('f'):
                self.roll += 5
            elif key in [ord('+'), ord('=')]:
                self.scale = min(2.0, self.scale + 0.1)
            elif key in [ord('-'), ord('_')]:
                self.scale = max(0.3, self.scale - 0.1)
            elif key == ord('0'):
                self.yaw = self.pitch = self.roll = 0
                self.scale = 1.0
                print("Orientation reset")
            elif key == ord('m'):
                modes = ['pca', 'segment', 'both']
                self.mode = modes[(modes.index(self.mode) + 1) % len(modes)]
                print(f"Mode: {self.mode}")
            elif key == ord('s'):
                self.smooth = not self.smooth
            elif key == ord('p'):
                self.show_patch = not self.show_patch
            elif key == ord('o'):
                # Re-train
                frames = collect_frames(cap, self.transform, self.device, self.args.n_frames)
                eot_config = EOTConfig(n_transforms=self.args.n_transforms)
                attack = EOTAttack(self.model, eot_config, self.device)
                self.patch = attack.optimize(frames, self.args.patch_size, self.args.steps, self.args.lr)
                self.patch_np = (self.patch.cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                self.patch_np = cv2.cvtColor(self.patch_np, cv2.COLOR_RGB2BGR)
            elif key == ord('w'):
                save_path = Path(__file__).parent.parent / "results" / f"eot_patch_{int(time.time())}.pt"
                torch.save(self.patch, str(save_path))
                print(f"Saved to {save_path}")

        cap.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='EOT Interactive Attack')
    parser.add_argument('--size', type=int, default=400, help='Display size')
    parser.add_argument('--patch-size', type=int, default=50, help='Patch size')
    parser.add_argument('--mode', choices=['pca', 'segment', 'both'], default='segment')
    parser.add_argument('--clusters', type=int, default=4)
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--n-frames', type=int, default=30)
    parser.add_argument('--n-transforms', type=int, default=8, help='Transforms per step')
    parser.add_argument('--steps', type=int, default=400)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--load-patch', type=str, default=None)
    parser.add_argument('--retrain', action='store_true', help='Force retrain')

    args = parser.parse_args()
    app = EOTInteractiveDemo(args)
    app.run()


if __name__ == "__main__":
    main()
