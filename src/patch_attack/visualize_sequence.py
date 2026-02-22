from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from PIL import Image
from sklearn.decomposition import PCA
from torchvision import transforms
from tqdm import tqdm

from .models.dinov3_loader import load_dinov3
from .utils.config import (
    CLASSIFIER,
    FOCUS_CLASSES,
    FPS,
    PATCH,
    PATCH_MIN_ROW_RATIO,
    PATCH_PERSPECTIVE_MIN_SCALE,
    PATCH_SIZE,
    PATCH_Y_RATIO,
    REFRESH,
    SOURCE_CLASS,
    VIZ_DATASET,
    VIZ_SEQ_SIZE,
    get_device,
)
from .utils.viz import (
    CITYSCAPES_COLORS,
    CLASS_NAMES,
    OTHER_COLOR,
    compute_perspective_size,
    create_legend,
)

DEMO_DIR = Path("results/demo")


def apply_patch(image: torch.Tensor, patch: torch.Tensor, pos: tuple) -> torch.Tensor:
    x, y = pos
    out = image.clone()
    xe = min(x + patch.shape[1], image.shape[1])
    ye = min(y + patch.shape[2], image.shape[2])
    out[:, x:xe, y:ye] = patch[:, : xe - x, : ye - y]
    return out


def resize_patch(patch: torch.Tensor, size: int) -> torch.Tensor:
    result: torch.Tensor = F.interpolate(
        patch.unsqueeze(0), (size, size), mode="bilinear", align_corners=False
    ).squeeze(0)
    return result


def pca_scatter(
    token_sets: list, pca_model: PCA | None, width: int, height: int, n_samples: int = 40
) -> tuple:
    from mpl_toolkits.mplot3d import Axes3D  # noqa

    if pca_model is None:
        pca_model = PCA(n_components=3).fit(np.vstack([t for _, t, _ in token_sets]))
    projected = [(lbl, pca_model.transform(t), c) for lbl, t, c in token_sets]
    n_total = len(projected[0][1])
    idx = np.round(np.linspace(0, n_total - 1, min(n_samples, n_total))).astype(int)
    fig = Figure(figsize=(width / 100, height / 100), dpi=100)
    fig.patch.set_facecolor("#0f0f1a")
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#0f0f1a")
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = False
        pane.set_edgecolor("#222233")
    ax.grid(True, alpha=0.08)
    ax.view_init(elev=20, azim=45)
    for lbl, pts, color in projected:
        s = pts[idx]
        ax.scatter(
            s[:, 0],
            s[:, 1],
            s[:, 2],
            c=color,
            s=18,
            alpha=0.85,
            label=lbl,
            depthshade=True,
            linewidths=0,
        )
    ax.set_xlabel("PC1", color="#888", fontsize=6, labelpad=2)
    ax.set_ylabel("PC2", color="#888", fontsize=6, labelpad=2)
    ax.set_zlabel("PC3", color="#888", fontsize=6, labelpad=2)
    ax.tick_params(colors="#777", labelsize=4)
    ax.legend(
        fontsize=7,
        facecolor="#1a1a2e",
        edgecolor="#444466",
        labelcolor="white",
        loc="upper left",
        markerscale=1.5,
        framealpha=0.8,
    )
    ax.set_title("Embeddings DINOv3 — PCA 3D", color="#cccccc", fontsize=8, pad=6)
    fig.tight_layout(pad=0.3)
    canvas.draw()
    img = np.asarray(canvas.buffer_rgba())[:, :, :3].copy()
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR), pca_model


class SequenceVisualizer:
    def __init__(self) -> None:
        self.device = get_device()
        print("Chargement du modèle...")
        self.model = load_dinov3(self.device)
        clf_data = torch.load(CLASSIFIER, map_location=self.device, weights_only=False)
        self.clf = torch.nn.Linear(384, 19).to(self.device)
        self.clf.load_state_dict(clf_data["state_dict"])
        self.clf.eval()
        self.img_size: int = clf_data.get("img_size", 224)
        self.grid = self.img_size // 16
        self.patch = torch.load(PATCH, map_location=self.device, weights_only=True).to(self.device)
        if self.patch.shape[1] != PATCH_SIZE or self.patch.shape[2] != PATCH_SIZE:
            self.patch = resize_patch(self.patch, PATCH_SIZE)
        print(
            f"Patch : {self.patch.shape} | Image {self.img_size}×{self.img_size} "
            f"→ {self.grid}×{self.grid} tokens | {self.device}"
        )
        self.cfgs = self._dist_configs()
        self.transform = transforms.Compose(
            [
                transforms.Resize(self.img_size + 32),
                transforms.CenterCrop(self.img_size),
                transforms.ToTensor(),
            ]
        )
        self.src_name = CLASS_NAMES[SOURCE_CLASS]
        print("Configurations distance :")
        for c in self.cfgs:
            print(f"  {c['name']:8s}: row={c['x']}, col={c['y']}, taille={c['size']}px")

    def _dist_configs(self) -> list[dict]:
        configs = []
        for name, ratio in [
            ("Loin", PATCH_MIN_ROW_RATIO),
            ("Moyen", PATCH_MIN_ROW_RATIO + (0.78 - PATCH_MIN_ROW_RATIO) * 0.45),
            ("Proche", PATCH_MIN_ROW_RATIO + (0.78 - PATCH_MIN_ROW_RATIO) * 0.90),
        ]:
            x = int(self.img_size * ratio)
            eff = compute_perspective_size(
                x, self.img_size, PATCH_SIZE, PATCH_PERSPECTIVE_MIN_SCALE
            )
            y = min(int((self.img_size - eff) * PATCH_Y_RATIO), self.img_size - eff - 1)
            configs.append({"name": name, "x": x, "y": y, "size": eff})
        return configs

    def get_tokens(self, img_tensor: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            feats = self.model.get_intermediate_layers(img_tensor.unsqueeze(0), n=1)[0]
        return F.normalize(feats[0, 1:], dim=1)

    def clf_vis(self, tokens: torch.Tensor, size: int) -> tuple[np.ndarray, np.ndarray]:
        with torch.no_grad():
            preds = self.clf(tokens).argmax(-1).cpu().numpy()
        n = self.grid * self.grid
        preds = np.concatenate([preds, np.full(max(0, n - len(preds)), preds[-1])])[:n]
        seg = preds.reshape(self.grid, self.grid)
        colored = np.full((self.grid, self.grid, 3), OTHER_COLOR, dtype=np.uint8)
        for c in FOCUS_CLASSES:
            colored[seg == c] = CITYSCAPES_COLORS[c]
        return cv2.resize(colored, (size, size), interpolation=cv2.INTER_NEAREST), preds

    def process_frame(
        self, img_tensor: torch.Tensor, size: int
    ) -> tuple[list, list, int, list, np.ndarray, list]:
        tokens_ref = self.get_tokens(img_tensor)
        ref_vis, ref_preds = self.clf_vis(tokens_ref, size)
        ref_bgr = cv2.cvtColor(ref_vis, cv2.COLOR_RGB2BGR)
        n_src = int((ref_preds == SOURCE_CLASS).sum())
        cv2.putText(
            ref_bgr, "Original", (5, size - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1
        )
        if n_src == 0:
            cv2.putText(
                ref_bgr,
                f"Aucun {self.src_name}",
                (5, size // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (150, 150, 150),
                1,
            )
        panels = [ref_bgr]
        heatmaps = []
        token_sets = [("Clean", tokens_ref.cpu().numpy(), "#4fc3f7")]
        metrics: list[tuple[float, bool]] = []
        _colors = ["#1E88E5", "#FF9800", "#E53935"]

        for di, cfg in enumerate(self.cfgs):
            patch_d = resize_patch(self.patch, cfg["size"])
            patched = apply_patch(img_tensor, patch_d, (cfg["x"], cfg["y"]))
            tokens_d = self.get_tokens(patched)
            adv_vis, adv_preds = self.clf_vis(tokens_d, size)
            adv_bgr = cv2.cvtColor(adv_vis, cv2.COLOR_RGB2BGR)
            n_adv = int((adv_preds == SOURCE_CLASS).sum())
            fr = max(0.0, min(1.0, (n_src - n_adv) / n_src if n_src > 0 else 0.0))
            gone = n_adv == 0 and n_src > 0
            metrics.append((fr, gone))
            if gone:
                cv2.rectangle(adv_bgr, (0, 0), (size - 1, size - 1), (0, 0, 255), 4)
                cv2.putText(
                    adv_bgr,
                    "DISPARU!",
                    (size // 5, size // 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 0, 255),
                    2,
                )
            else:
                g, r = int(200 * fr), int(200 * (1 - fr))
                cv2.putText(
                    adv_bgr,
                    f"FR: {fr:.0%}",
                    (8, size // 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, g, r),
                    2,
                )
            cv2.putText(
                adv_bgr,
                f"{cfg['name']} ({cfg['size']}px)",
                (5, size - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (200, 200, 200),
                1,
            )
            panels.append(adv_bgr)
            token_sets.append((cfg["name"], tokens_d.cpu().numpy(), _colors[di]))
            dists = torch.norm(tokens_d - tokens_ref, dim=1).cpu().numpy()
            n_p = self.grid * self.grid
            dists = np.concatenate([dists, np.zeros(max(0, n_p - len(dists)))])[:n_p]
            dm = dists.reshape(self.grid, self.grid)
            dm = cv2.resize(dm, (size, size), interpolation=cv2.INTER_NEAREST)
            dm = (dm / (dm.max() + 1e-8) * 255).astype(np.uint8)
            heatmaps.append(cv2.applyColorMap(dm, cv2.COLORMAP_HOT))

        return panels, heatmaps, n_src, token_sets, ref_preds, metrics

    def build_frame(
        self,
        panels: list,
        heatmaps: list,
        scatter: np.ndarray,
        legend: np.ndarray,
        n_src: int,
        frame_idx: int,
        img_path: Path,
        size: int,
    ) -> np.ndarray:
        empty = np.full((size, size // 2, 3), 30, dtype=np.uint8)
        row1 = np.hstack(panels + [legend])
        row2 = np.hstack([scatter] + heatmaps + [empty])
        frame = np.vstack([row1, row2])
        cv2.putText(
            frame, "Original", (5, size - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1
        )
        for i, cfg in enumerate(self.cfgs):
            cv2.putText(
                frame,
                cfg["name"],
                ((i + 1) * size + 5, size - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (200, 200, 200),
                1,
            )
        cv2.putText(
            frame, "PCA 3D", (5, 2 * size - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1
        )
        for i, cfg in enumerate(self.cfgs):
            cv2.putText(
                frame,
                f"Perturb. {cfg['name']}",
                ((i + 1) * size + 5, 2 * size - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (200, 200, 200),
                1,
            )
        cv2.putText(
            frame,
            f"{self.src_name}: {n_src} tokens | frame {frame_idx + 1}",
            (10, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )
        cv2.putText(
            frame, img_path.name, (10, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1
        )
        return frame

    def save_analysis(self, fooling_history: list, disappeared_frames: list) -> None:
        if not fooling_history or not len(fooling_history[0]):
            return
        n_frames = len(fooling_history[0])
        x = np.arange(n_frames)
        plot_colors = ["#2196F3", "#FF9800", "#F44336"]
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(14, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
        )
        for di, cfg in enumerate(self.cfgs):
            ax1.plot(
                x,
                fooling_history[di],
                color=plot_colors[di],
                lw=2,
                label=f"{cfg['name']} ({cfg['size']}px)",
            )
            if disappeared_frames[di]:
                ax1.scatter(
                    disappeared_frames[di],
                    [fooling_history[di][f] for f in disappeared_frames[di]],
                    color=plot_colors[di],
                    marker="v",
                    s=100,
                    zorder=5,
                )
        ax1.axhline(1.0, color="gray", ls="--", alpha=0.4, lw=1)
        ax1.set_ylabel("Taux de tromperie")
        ax1.set_ylim(-0.05, 1.15)
        ax1.set_title(f"Erreur de perception — '{self.src_name}' — 3 distances (▼ = disparition)")
        ax1.legend(loc="upper right")
        ax1.grid(True, alpha=0.3)
        for di, cfg in enumerate(self.cfgs):
            gone_set = set(disappeared_frames[di])
            signal = np.array([1 if f in gone_set else 0 for f in range(n_frames)], dtype=float)
            pct = 100 * len(disappeared_frames[di]) // max(1, n_frames)
            ax2.fill_between(
                x,
                signal,
                step="mid",
                alpha=0.65,
                color=plot_colors[di],
                label=f"{cfg['name']}: {len(disappeared_frames[di])} frames ({pct}%)",
            )
        ax2.set_ylabel("Disparu")
        ax2.set_xlabel("Frame")
        ax2.set_ylim(0, 1.5)
        ax2.set_yticks([])
        ax2.legend(loc="upper right", fontsize=9)
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        plot_path = DEMO_DIR / (Path(VIZ_DATASET).name + "_analysis.png")
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        print(f"Graphe → {plot_path}")
        plt.show()

    def run(self) -> None:
        image_paths = sorted(
            p for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp") for p in Path(VIZ_DATASET).glob(ext)
        )
        print(f"{len(image_paths)} images dans {VIZ_DATASET}")
        if not image_paths:
            return
        size = VIZ_SEQ_SIZE
        width = size * 4 + size // 2
        height = size * 2
        DEMO_DIR.mkdir(parents=True, exist_ok=True)
        out_path = DEMO_DIR / (Path(VIZ_DATASET).name + ".mp4")
        fourcc: int = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
        writer = cv2.VideoWriter(str(out_path), fourcc, FPS, (width, height))
        print(f"Vidéo → {out_path}")
        fooling_history: list[list[float]] = [[] for _ in self.cfgs]
        disappeared_frames: list[list[int]] = [[] for _ in self.cfgs]
        global_pca: PCA | None = None
        cv2.namedWindow("Patch Attack Visualization", cv2.WINDOW_NORMAL)

        for frame_idx, img_path in enumerate(tqdm(image_paths)):
            img_tensor = self.transform(Image.open(img_path).convert("RGB")).to(self.device)
            if REFRESH > 0 and frame_idx % REFRESH == 0:
                global_pca = None
            panels, heatmaps, n_src, token_sets, ref_preds, metrics = self.process_frame(
                img_tensor, size
            )
            for di, (fr, gone) in enumerate(metrics):
                fooling_history[di].append(fr)
                if gone:
                    disappeared_frames[di].append(len(fooling_history[di]) - 1)
            legend = create_legend(size, np.unique(ref_preds), focus_classes=FOCUS_CLASSES)
            scatter, global_pca = pca_scatter(token_sets, global_pca, size, size)
            frame = self.build_frame(
                panels, heatmaps, scatter, legend, n_src, frame_idx, img_path, size
            )
            writer.write(frame)
            cv2.imshow("Patch Attack Visualization", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord(" "):
                cv2.waitKey(0)

        writer.release()
        cv2.destroyAllWindows()
        print(f"\nVidéo sauvegardée → {out_path}")
        self.save_analysis(fooling_history, disappeared_frames)


def main() -> None:
    SequenceVisualizer().run()


if __name__ == "__main__":
    main()
