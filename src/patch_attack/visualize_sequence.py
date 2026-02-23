import random
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
    token_sets: list,
    pca_model: PCA | None,
    width: int,
    height: int,
    n_samples: int = 80,
    draw_lines: bool = True,
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
    if draw_lines and len(projected) >= 2:
        pts0 = projected[0][1][idx]
        pts1 = projected[1][1][idx]
        for p0, p1 in zip(pts0, pts1):
            ax.plot(
                [p0[0], p1[0]],
                [p0[1], p1[1]],
                [p0[2], p1[2]],
                color="white",
                alpha=0.45,
                lw=0.8,
            )
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
        self.viz_cfg = self._random_viz_cfg()
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
        print(
            f"Position visu (aléatoire) : row={self.viz_cfg['x']}, col={self.viz_cfg['y']}, "
            f"taille={self.viz_cfg['size']}px"
        )

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

    def _random_viz_cfg(self) -> dict:
        row_ratio = random.uniform(PATCH_MIN_ROW_RATIO, 0.78)
        y_ratio = random.uniform(PATCH_Y_RATIO, min(PATCH_Y_RATIO + 0.10, 0.97))
        x = int(self.img_size * row_ratio)
        eff = compute_perspective_size(x, self.img_size, PATCH_SIZE, PATCH_PERSPECTIVE_MIN_SCALE)
        y = min(int((self.img_size - eff) * y_ratio), self.img_size - eff - 1)
        return {"name": "visu", "x": x, "y": y, "size": eff}

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
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        list,
        int,
        np.ndarray,
        float,
        bool,
    ]:
        tokens_ref = self.get_tokens(img_tensor)
        ref_vis, ref_preds = self.clf_vis(tokens_ref, size)
        ref_bgr = cv2.cvtColor(ref_vis, cv2.COLOR_RGB2BGR)
        n_src = int((ref_preds == SOURCE_CLASS).sum())
        cv2.putText(
            ref_bgr,
            "Seg Original",
            (5, size - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (200, 200, 200),
            1,
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

        ref_preds_2d = ref_preds.reshape(self.grid, self.grid)
        cfg = self.viz_cfg
        patch_d = resize_patch(self.patch, cfg["size"])
        patched = apply_patch(img_tensor, patch_d, (cfg["x"], cfg["y"]))
        tokens_d = self.get_tokens(patched)
        adv_vis, adv_preds = self.clf_vis(tokens_d, size)
        n_adv = int((adv_preds == SOURCE_CLASS).sum())
        n_fooled = int(((ref_preds == SOURCE_CLASS) & (adv_preds != SOURCE_CLASS)).sum())
        fr = n_fooled / n_src if n_src > 0 else 0.0
        gone = n_adv == 0 and n_src > 0

        # Image + patch panel
        patched_np = (patched.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        img_panel = cv2.resize(cv2.cvtColor(patched_np, cv2.COLOR_RGB2BGR), (size, size))
        # Green rectangle showing patch position (perspective-scaled)
        scale = size / self.img_size
        px = int(cfg["y"] * scale)
        py = int(cfg["x"] * scale)
        ps = int(cfg["size"] * scale)
        cv2.rectangle(img_panel, (px, py), (px + ps, py + ps), (0, 255, 0), 2)
        cv2.putText(
            img_panel,
            "Image + Patch",
            (5, size - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (200, 200, 200),
            1,
        )
        # Attacked segmentation panel
        adv_bgr = cv2.cvtColor(adv_vis, cv2.COLOR_RGB2BGR)
        cv2.putText(
            adv_bgr,
            "Seg Attacked",
            (5, size - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (200, 200, 200),
            1,
        )
        # Semantic diff: green = source tokens surviving, red = source tokens fooled
        adv_preds_2d = adv_preds.reshape(self.grid, self.grid)
        diff = np.full((self.grid, self.grid, 3), OTHER_COLOR, dtype=np.uint8)
        for c in FOCUS_CLASSES:
            if c != SOURCE_CLASS:
                diff[ref_preds_2d == c] = CITYSCAPES_COLORS[c]
        diff[(ref_preds_2d == SOURCE_CLASS) & (adv_preds_2d == SOURCE_CLASS)] = [30, 200, 30]
        diff[(ref_preds_2d == SOURCE_CLASS) & (adv_preds_2d != SOURCE_CLASS)] = [220, 30, 30]
        diff = cv2.resize(diff, (size, size), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
        diff_panel = cv2.cvtColor(diff, cv2.COLOR_RGB2BGR)
        if gone:
            cv2.rectangle(diff_panel, (0, 0), (size - 1, size - 1), (0, 0, 255), 4)
            cv2.putText(
                diff_panel,
                "DISPARU!",
                (size // 5, size // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                2,
            )
        cv2.putText(
            diff_panel,
            f"{self.src_name} → ?",
            (5, size - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (200, 200, 200),
            1,
        )
        # L2 heatmap
        dists = torch.norm(tokens_d - tokens_ref, dim=1).cpu().numpy()
        n_p = self.grid * self.grid
        dists = np.concatenate([dists, np.zeros(max(0, n_p - len(dists)))])[:n_p]
        dm = dists.reshape(self.grid, self.grid)
        dm = cv2.resize(dm, (size, size), interpolation=cv2.INTER_NEAREST)
        dm = (dm / (dm.max() + 1e-8) * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(dm, cv2.COLORMAP_HOT)
        token_sets = [
            ("Clean", tokens_ref.cpu().numpy(), "#4fc3f7"),
            ("Attaqué", tokens_d.cpu().numpy(), "#ef5350"),
        ]
        return (
            img_panel,
            ref_bgr,
            adv_bgr,
            diff_panel,
            heatmap,
            token_sets,
            n_src,
            ref_preds,
            fr,
            gone,
        )

    def build_frame(
        self,
        img_panel: np.ndarray,
        ref_panel: np.ndarray,
        adv_panel: np.ndarray,
        diff_panel: np.ndarray,
        heatmap: np.ndarray,
        scatter: np.ndarray,
        legend: np.ndarray,
        n_src: int,
        fr: float,
        frame_idx: int,
        img_path: Path,
        size: int,
    ) -> np.ndarray:
        g, r = int(200 * fr), int(200 * (1 - fr))
        cv2.putText(
            legend,
            f"FR: {fr:.0%}",
            (10, size - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, g, r),
            2,
        )
        empty = np.full((size, size // 2, 3), 30, dtype=np.uint8)
        row1 = np.hstack([img_panel, ref_panel, adv_panel, diff_panel, legend])
        row2 = np.hstack([scatter, heatmap, empty])
        frame = np.vstack([row1, row2])
        cv2.putText(
            frame,
            "Embeddings DINOv3 — PCA 3D",
            (5, 2 * size - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (200, 200, 200),
            1,
        )
        cv2.putText(
            frame,
            "Perturbation L2",
            (3 * size + 5, 2 * size - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (200, 200, 200),
            1,
        )
        cv2.putText(
            frame,
            f"Fooling: {n_src} tokens | frame {frame_idx + 1}",
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

    def save_analysis(self, fooling_history: list[float], disappeared_frames: list[int]) -> None:
        if not fooling_history:
            return
        cfg = self.cfgs[1]  # Moyen — medium distance
        n_frames = len(fooling_history)
        x = np.arange(n_frames)
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(14, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
        )
        ax1.plot(
            x, fooling_history, color="#2196F3", lw=2, label=f"{cfg['name']} ({cfg['size']}px)"
        )
        if disappeared_frames:
            ax1.scatter(
                disappeared_frames,
                [fooling_history[f] for f in disappeared_frames],
                color="#2196F3",
                marker="v",
                s=100,
                zorder=5,
            )
        ax1.axhline(1.0, color="gray", ls="--", alpha=0.4, lw=1)
        ax1.set_ylabel("Fooling rate")
        ax1.set_ylim(-0.05, 1.15)
        ax1.set_title(f"Adversarial patch — '{self.src_name}' (▼ = disappeared)")
        ax1.legend(loc="upper right")
        ax1.grid(True, alpha=0.3)
        gone_set = set(disappeared_frames)
        signal = np.array([1 if f in gone_set else 0 for f in range(n_frames)], dtype=float)
        pct = 100 * len(disappeared_frames) // max(1, n_frames)
        ax2.fill_between(
            x,
            signal,
            step="mid",
            alpha=0.65,
            color="#2196F3",
            label=f"{len(disappeared_frames)} frames disappeared ({pct}%)",
        )
        ax2.set_ylabel("Gone")
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
        width = size * 4 + size // 2  # img+patch | seg orig | seg atk | diff | legend
        height = size * 2  # row1: panels  row2: pca(3×) + heatmap + empty
        DEMO_DIR.mkdir(parents=True, exist_ok=True)
        out_path = DEMO_DIR / (Path(VIZ_DATASET).name + ".mp4")
        fourcc: int = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
        writer = cv2.VideoWriter(str(out_path), fourcc, FPS, (width, height))
        print(f"Vidéo → {out_path}")
        fooling_history: list[float] = []
        disappeared_frames: list[int] = []
        global_pca: PCA | None = None
        cv2.namedWindow("Patch Attack Visualization", cv2.WINDOW_NORMAL)

        for frame_idx, img_path in enumerate(tqdm(image_paths)):
            img_tensor = self.transform(Image.open(img_path).convert("RGB")).to(self.device)
            if REFRESH > 0 and frame_idx % REFRESH == 0:
                global_pca = None
            (
                img_panel,
                ref_panel,
                adv_panel,
                diff_panel,
                heatmap,
                token_sets,
                n_src,
                ref_preds,
                fr,
                gone,
            ) = self.process_frame(img_tensor, size)
            fooling_history.append(fr)
            if gone:
                disappeared_frames.append(len(fooling_history) - 1)
            legend = create_legend(size, np.unique(ref_preds), focus_classes=FOCUS_CLASSES)
            scatter, global_pca = pca_scatter(
                token_sets, global_pca, 3 * size, size, draw_lines=True
            )
            frame = self.build_frame(
                img_panel,
                ref_panel,
                adv_panel,
                diff_panel,
                heatmap,
                scatter,
                legend,
                n_src,
                fr,
                frame_idx,
                img_path,
                size,
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
