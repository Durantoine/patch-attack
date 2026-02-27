from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from .utils.config import (
    IMG_SIZE,
    PATCH,
    PATCH_PERSPECTIVE_MIN_SCALE,
    PATCH_SIZE,
    VIZ_DATASET,
    VIZ_SEQ_SIZE,
    YOLO_CONF,
    YOLO_MODEL,
    YOLO_PERSON_CLASS,
    get_device,
)
from .utils.viz import (
    apply_patch,
    compute_perspective_size,
    letterbox,
    patch_to_img,
    resize_patch,
    tensor_to_bgr,
)

COLOR_CLEAN: tuple[int, int, int] = (0, 200, 80)
COLOR_ATTACKED: tuple[int, int, int] = (80, 120, 255)


@dataclass
class TransferMetrics:
    n_images: int = 0
    images_with_person: int = 0
    disappeared: int = 0
    total_clean: int = 0
    total_attacked: int = 0
    conf_clean: list[float] = field(default_factory=list)
    conf_attacked: list[float] = field(default_factory=list)
    det_clean_history: list[int] = field(default_factory=list)
    det_attacked_history: list[int] = field(default_factory=list)
    disappeared_flags: list[bool] = field(default_factory=list)

    @property
    def avg_det_clean(self) -> float:
        return self.total_clean / max(1, self.n_images)

    @property
    def avg_det_attacked(self) -> float:
        return self.total_attacked / max(1, self.n_images)

    @property
    def detection_drop(self) -> float:
        return (self.total_clean - self.total_attacked) / max(1, self.total_clean)

    @property
    def disappearance_rate(self) -> float:
        return self.disappeared / max(1, self.images_with_person)

    @property
    def avg_conf_clean(self) -> float:
        return float(np.mean(self.conf_clean)) if self.conf_clean else 0.0

    @property
    def avg_conf_attacked(self) -> float:
        return float(np.mean(self.conf_attacked)) if self.conf_attacked else 0.0

    @property
    def avg_conf_drop(self) -> float:
        return self.avg_conf_clean - self.avg_conf_attacked


class TransferEvaluator:
    def __init__(self) -> None:
        self.device = get_device()
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise ImportError("ultralytics requis : pip install ultralytics") from e
        print(f"Chargement YOLO ({YOLO_MODEL})...")
        self.yolo = YOLO(YOLO_MODEL)
        patch_raw: torch.Tensor = torch.load(PATCH, map_location="cpu", weights_only=True)
        if patch_raw.ndim == 4:
            patch_raw = patch_raw[0]
        self.patch = patch_raw.float().clamp(0, 1).to(self.device)
        print(f"Patch chargé : {self.patch.shape} depuis {PATCH}")

    def _patch_position(self, H: int) -> tuple[int, int, int]:
        x = int(H * 0.65)
        eff = compute_perspective_size(x, H, PATCH_SIZE, PATCH_PERSPECTIVE_MIN_SCALE)
        y = min(int((H - eff) * 0.925), H - eff - 1)
        return x, y, eff

    def _detect_persons(self, img_bgr: np.ndarray) -> list[dict]:
        results = self.yolo(img_bgr, verbose=False, conf=YOLO_CONF)[0]
        return [
            {"bbox": tuple(int(v) for v in box.xyxy[0].tolist()), "conf": float(box.conf.item())}
            for box in results.boxes
            if int(box.cls.item()) == YOLO_PERSON_CLASS
        ]

    def _draw_detections(
        self, img_bgr: np.ndarray, detections: list[dict], color: tuple[int, int, int]
    ) -> np.ndarray:
        out = img_bgr.copy()
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            cv2.putText(out, f"{det['conf']:.2f}", (x1, max(y1 - 5, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        return out

    def _summary_panel(self, metrics: TransferMetrics, size: int) -> np.ndarray:
        panel = np.full((size, size, 3), 25, dtype=np.uint8)
        lines: list[tuple[str, tuple[int, int, int], float]] = [
            ("Transferabilite",                        (200, 200, 200), 0.55),
            ("DINOv3 -> YOLO",                         (160, 160, 160), 0.45),
            ("",                                       (0, 0, 0),       0.0),
            (f"Images: {metrics.n_images}",            (200, 200, 200), 0.5),
            ("",                                       (0, 0, 0),       0.0),
            ("Detections moy.:",                       (200, 200, 200), 0.5),
            (f"  Clean:   {metrics.avg_det_clean:.1f}/img",    COLOR_CLEAN,    0.5),
            (f"  Attaque: {metrics.avg_det_attacked:.1f}/img", COLOR_ATTACKED, 0.5),
            (f"  Chute:   {metrics.detection_drop:.1%}",       (0, 220, 255),  0.5),
            ("",                                       (0, 0, 0),       0.0),
            ("Taux disparition:",                      (200, 200, 200), 0.5),
            (f"  {metrics.disappearance_rate:.1%}",    (0, 220, 255),   0.65),
            ("",                                       (0, 0, 0),       0.0),
            ("Conf. moy.:",                            (200, 200, 200), 0.5),
            (f"  Clean:   {metrics.avg_conf_clean:.3f}",    COLOR_CLEAN,    0.5),
            (f"  Attaque: {metrics.avg_conf_attacked:.3f}", COLOR_ATTACKED, 0.5),
            (f"  Chute:   {metrics.avg_conf_drop:.3f}",     (0, 220, 255),  0.5),
        ]
        y = 28
        for text, color, scale in lines:
            if not text:
                y += 8
                continue
            cv2.putText(panel, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1)
            y += int(scale * 38 + 6)
        return panel

    def _build_frame(
        self,
        disp_clean: np.ndarray,
        disp_attacked: np.ndarray,
        dets_clean: list[dict],
        dets_attacked: list[dict],
        metrics: TransferMetrics,
        size: int,
        lb_scale: float,
        lb_px: int,
        lb_py: int,
    ) -> np.ndarray:
        def _rescale(dets: list[dict]) -> list[dict]:
            return [
                {"bbox": (int(d["bbox"][0] * lb_scale) + lb_px,
                          int(d["bbox"][1] * lb_scale) + lb_py,
                          int(d["bbox"][2] * lb_scale) + lb_px,
                          int(d["bbox"][3] * lb_scale) + lb_py),
                 "conf": d["conf"]}
                for d in dets
            ]

        panel_clean = self._draw_detections(disp_clean, _rescale(dets_clean), COLOR_CLEAN)
        panel_attacked = self._draw_detections(disp_attacked, _rescale(dets_attacked), COLOR_ATTACKED)
        for panel, label in [(panel_clean, f"Clean ({len(dets_clean)} pers.)"),
                              (panel_attacked, f"Attaque ({len(dets_attacked)} pers.)")]:
            cv2.putText(panel, label, (5, size - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        patch_panel = patch_to_img(self.patch, size)
        cv2.putText(patch_panel, "Patch", (5, size - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        return np.hstack([panel_clean, panel_attacked, patch_panel, self._summary_panel(metrics, size)])

    def _update_metrics(
        self, metrics: TransferMetrics, dets_clean: list[dict], dets_attacked: list[dict]
    ) -> None:
        gone = bool(dets_clean and not dets_attacked)
        metrics.n_images += 1
        metrics.total_clean += len(dets_clean)
        metrics.total_attacked += len(dets_attacked)
        metrics.det_clean_history.append(len(dets_clean))
        metrics.det_attacked_history.append(len(dets_attacked))
        metrics.disappeared_flags.append(gone)
        if dets_clean:
            metrics.images_with_person += 1
            if gone:
                metrics.disappeared += 1
            metrics.conf_clean.append(max(d["conf"] for d in dets_clean))
            metrics.conf_attacked.append(max((d["conf"] for d in dets_attacked), default=0.0))

    def run(self) -> TransferMetrics:
        tf = transforms.Compose([
            transforms.Resize(IMG_SIZE + 32),
            transforms.CenterCrop(IMG_SIZE),
            transforms.ToTensor(),
        ])
        paths = sorted(
            p for ext in ("*.png", "*.jpg", "*.jpeg")
            for p in Path(VIZ_DATASET).resolve().glob(f"**/{ext}")
        )
        print(f"{len(paths)} images dans {VIZ_DATASET}")
        if not paths:
            return TransferMetrics()

        cv2.namedWindow("Transferability Eval", cv2.WINDOW_NORMAL)
        metrics = TransferMetrics()
        for img_path in tqdm(paths, desc="Évaluation transfert"):
            img_t = tf(Image.open(img_path).convert("RGB")).to(self.device)
            x, y, eff = self._patch_position(img_t.shape[1])
            patched_t = apply_patch(img_t, resize_patch(self.patch, eff), (x, y))

            clean_bgr = tensor_to_bgr(img_t)
            patched_bgr = tensor_to_bgr(patched_t)
            dets_clean = self._detect_persons(clean_bgr)
            dets_attacked = self._detect_persons(patched_bgr)
            self._update_metrics(metrics, dets_clean, dets_attacked)

            disp_clean, lb_scale, lb_px, lb_py = letterbox(clean_bgr, VIZ_SEQ_SIZE)
            disp_attacked, _, _, _ = letterbox(patched_bgr, VIZ_SEQ_SIZE)
            frame = self._build_frame(disp_clean, disp_attacked, dets_clean, dets_attacked,
                                      metrics, VIZ_SEQ_SIZE, lb_scale, lb_px, lb_py)
            cv2.imshow("Transferability Eval", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cv2.destroyAllWindows()
        return metrics

    def plot_analysis(self, metrics: TransferMetrics) -> None:
        if not metrics.det_clean_history:
            return
        x = np.arange(len(metrics.det_clean_history))
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(14, 6), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
        )
        ax1.plot(x, metrics.det_clean_history, color="#4caf50", lw=1.5, label="Clean")
        ax1.plot(x, metrics.det_attacked_history, color="#2196f3", lw=1.5, label="Attaqué")
        ax1.fill_between(x, metrics.det_clean_history, metrics.det_attacked_history,
                         alpha=0.15, color="#2196f3")
        ax1.set_ylabel("Détections YOLO (person)")
        ax1.set_title(f"Transférabilité DINOv3 → YOLO | {Path(VIZ_DATASET).name} "
                      f"| taux disparition : {metrics.disappearance_rate:.1%}")
        ax1.legend(loc="upper right")
        ax1.grid(True, alpha=0.3)

        gone_signal = np.array(metrics.disappeared_flags, dtype=float)
        pct = 100 * metrics.disappeared // max(1, metrics.images_with_person)
        ax2.fill_between(x, gone_signal, step="mid", alpha=0.65, color="#2196f3",
                         label=f"{metrics.disappeared} disparues ({pct}%)")
        ax2.set_ylabel("Disparu")
        ax2.set_xlabel("Image")
        ax2.set_ylim(0, 1.5)
        ax2.set_yticks([])
        ax2.legend(loc="upper right", fontsize=9)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        out = Path("results") / f"transfer_{Path(VIZ_DATASET).name}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Graphe → {out}")
        plt.show()

    def report(self, metrics: TransferMetrics) -> None:
        print("\n" + "=" * 52)
        print("RÉSULTATS TRANSFÉRABILITÉ (DINOv3 patch → YOLO)")
        print("=" * 52)
        print(f"Dataset          : {VIZ_DATASET}")
        print(f"Patch            : {PATCH}")
        print(f"Images évaluées  : {metrics.n_images}")
        print(f"Images w/ person : {metrics.images_with_person}")
        print()
        print(f"Dét. moy. (clean)   : {metrics.avg_det_clean:.2f}/img")
        print(f"Dét. moy. (attaque) : {metrics.avg_det_attacked:.2f}/img")
        print(f"Chute détections    : {metrics.detection_drop:.1%}")
        print()
        print(f"Taux disparition    : {metrics.disappeared}/{metrics.images_with_person}"
              f" = {metrics.disappearance_rate:.1%}")
        print()
        print(f"Conf. moy. (clean)  : {metrics.avg_conf_clean:.4f}")
        print(f"Conf. moy. (attaque): {metrics.avg_conf_attacked:.4f}")
        print(f"Chute conf. moy.    : {metrics.avg_conf_drop:.4f}")
        print("=" * 52)


def main() -> None:
    evaluator = TransferEvaluator()
    metrics = evaluator.run()
    evaluator.plot_analysis(metrics)
    evaluator.report(metrics)


if __name__ == "__main__":
    main()
