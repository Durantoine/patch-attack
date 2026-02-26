"""Evaluate transferability of a DINOv3-trained adversarial patch to YOLOv8 detection.

Measures whether a patch optimized to fool DINOv3 segmentation also degrades
YOLO object detection — without any retraining.

Usage:
    python scripts/eval_transferability.py --dataset data/stuttgart_02 --patch results/targeted_patch_best.pt
    python scripts/eval_transferability.py --dataset data/stuttgart_02 --patch results/targeted_patch_best.pt --output results/transfer_eval.mp4
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import argparse

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from ultralytics import YOLO

from utils.config import (
    DATASET, IMG_SIZE, PATCH_SIZE, PATCH_PERSPECTIVE_MIN_SCALE,
    PATCH_MIN_ROW_RATIO, PATCH_Y_RATIO, VIZ_SEQ_SIZE,
)
from utils.viz import compute_perspective_size

# COCO class index for "person"
YOLO_PERSON_CLASS: int = 0

YOLO_COLORS: dict[str, tuple[int, int, int]] = {
    "clean":   (0, 200, 80),   # green
    "attacked": (0, 60, 220),  # red-ish
}


def get_device() -> torch.device:
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


def apply_patch(image: torch.Tensor, patch: torch.Tensor,
                patch_size: int, position: tuple[int, int]) -> torch.Tensor:
    """Apply patch on a single image tensor [3, H, W]. Returns new tensor."""
    out = image.clone()
    ps = patch_size
    x, y = position
    resized = F.interpolate(patch.unsqueeze(0), size=(ps, ps),
                            mode="bilinear", align_corners=False)[0]
    xe = min(x + ps, image.shape[1])
    ye = min(y + ps, image.shape[2])
    out[:, x:xe, y:ye] = resized[:, :xe - x, :ye - y]
    return out


def detect_persons(yolo: YOLO, img_np: np.ndarray,
                   conf_threshold: float) -> list[dict]:
    """Run YOLO on a BGR numpy image. Returns list of person detections."""
    results = yolo(img_np, verbose=False, conf=conf_threshold)[0]
    detections = []
    for box in results.boxes:
        if int(box.cls.item()) == YOLO_PERSON_CLASS:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append({
                "bbox": (int(x1), int(y1), int(x2), int(y2)),
                "conf": float(box.conf.item()),
            })
    return detections


def draw_detections(img_bgr: np.ndarray, detections: list[dict],
                    color: tuple[int, int, int], label_prefix: str = "") -> np.ndarray:
    """Draw bounding boxes on image."""
    out = img_bgr.copy()
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"{label_prefix}{det['conf']:.2f}"
        cv2.putText(out, label, (x1, max(y1 - 5, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    return out


def tensor_to_bgr(t: torch.Tensor) -> np.ndarray:
    """[3, H, W] float tensor in [0,1] -> BGR uint8."""
    arr = (t.detach().cpu().permute(1, 2, 0).numpy().clip(0, 1) * 255).astype(np.uint8)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def letterbox(img_bgr: np.ndarray, size: int) -> tuple[np.ndarray, float, int, int]:
    """Resize image to fit within (size x size) while preserving aspect ratio.

    Returns (resized_img, scale, pad_x, pad_y) where pad_x/pad_y are pixel offsets.
    """
    h, w = img_bgr.shape[:2]
    scale = min(size / w, size / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    pad_x = (size - new_w) // 2
    pad_y = (size - new_h) // 2
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
    return canvas, scale, pad_x, pad_y


def make_summary_panel(metrics: dict, width: int = 300) -> np.ndarray:
    """Create a dark panel showing aggregated metrics."""
    panel = np.zeros((width, width, 3), dtype=np.uint8)
    panel[:] = 25
    lines = [
        ("Transferability", (200, 200, 200), 0.55),
        ("DINOv3 -> YOLO", (160, 160, 160), 0.45),
        ("", None, 0),
        (f"Images: {metrics['n_images']}", (200, 200, 200), 0.5),
        (f"Source: {metrics['source_class']}", (200, 200, 200), 0.5),
        ("", None, 0),
        ("Person detections:", (200, 200, 200), 0.5),
        (f"  Clean:    {metrics['avg_det_clean']:.1f}/img", (0, 200, 80), 0.5),
        (f"  Attacked: {metrics['avg_det_attacked']:.1f}/img", (80, 120, 255), 0.5),
        ("", None, 0),
        ("Disappearance rate:", (200, 200, 200), 0.5),
        (f"  {metrics['disappearance_rate']:.1%}", (0, 220, 255), 0.65),
        ("", None, 0),
        ("Avg confidence:", (200, 200, 200), 0.5),
        (f"  Clean:    {metrics['avg_conf_clean']:.3f}", (0, 200, 80), 0.5),
        (f"  Attacked: {metrics['avg_conf_attacked']:.3f}", (80, 120, 255), 0.5),
        (f"  Drop:     {metrics['avg_conf_drop']:.3f}", (0, 220, 255), 0.5),
    ]
    y = 28
    for text, color, scale in lines:
        if not text:
            y += 8
            continue
        cv2.putText(panel, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, color or (200, 200, 200), 1)
        y += int(scale * 38 + 6)
    return panel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=DATASET)
    parser.add_argument("--patch", default="results/targeted_patch_best.pt")
    parser.add_argument("--patch-size", type=int, default=PATCH_SIZE)
    parser.add_argument("--patch-pos", type=int, nargs=2, default=None,
                        help="Fixed patch position row col. Default: auto (perspective-scaled).")
    parser.add_argument("--perspective-min-scale", type=float, default=PATCH_PERSPECTIVE_MIN_SCALE)
    parser.add_argument("--min-row-ratio", type=float, default=PATCH_MIN_ROW_RATIO)
    parser.add_argument("--patch-y-ratio", type=float, default=PATCH_Y_RATIO)
    parser.add_argument("--conf", type=float, default=0.25,
                        help="YOLO confidence threshold")
    parser.add_argument("--yolo-model", default="yolov8n.pt")
    parser.add_argument("--source-class", default="person",
                        help="Label shown in output (informational only)")
    parser.add_argument("--size", type=int, default=VIZ_SEQ_SIZE,
                        help="Display panel size in pixels")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--output", default=None,
                        help="Save result video to this path (.mp4)")
    parser.add_argument("--max-images", type=int, default=0,
                        help="Limit number of images (0 = all)")
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    # Load patch
    patch_raw: torch.Tensor = torch.load(args.patch, map_location="cpu", weights_only=True)
    # Ensure patch is [3, H, W] float in [0, 1]
    if patch_raw.ndim == 4:
        patch_raw = patch_raw[0]
    patch: torch.Tensor = patch_raw.float().clamp(0, 1).to(device)
    print(f"Patch loaded: {patch.shape} from {args.patch}")

    # Load YOLO
    print(f"Loading YOLO ({args.yolo_model})...")
    yolo = YOLO(args.yolo_model)

    # Collect images
    paths: list[Path] = sorted(
        p for ext in ["*.png", "*.jpg", "*.jpeg"]
        for p in Path(args.dataset).resolve().glob(f"**/{ext}") if p.is_file()
    )
    if args.max_images > 0:
        paths = paths[:args.max_images]
    print(f"{len(paths)} images in {args.dataset}")
    if not paths:
        print("No images found.")
        return

    # Same transform as attack_classifier.py — keeps patch/image ratio identical to training
    tf = transforms.Compose([
        transforms.Resize(IMG_SIZE + 32),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
    ])

    # Metrics accumulators
    total_clean: int = 0
    total_attacked: int = 0
    disappeared: int = 0          # images where detections dropped to 0 (were > 0 clean)
    images_with_person: int = 0
    conf_clean_list: list[float] = []
    conf_attacked_list: list[float] = []

    # Video writer
    sz = args.size
    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        frame_w = sz * 3 + sz  # clean | patched | patch | summary
        writer = cv2.VideoWriter(args.output, fourcc, args.fps, (frame_w, sz))
        if not writer.isOpened():
            print(f"Warning: could not open video writer for {args.output}")
            writer = None
    else:
        cv2.namedWindow("Transferability Eval", cv2.WINDOW_NORMAL)
        cv2.moveWindow("Transferability Eval", 100, 100)

    for img_path in tqdm(paths, desc="Evaluating"):
        pil_img = Image.open(img_path).convert("RGB")
        img_t: torch.Tensor = tf(pil_img).to(device)
        _, H, W = img_t.shape

        # Compute patch position
        if args.patch_pos is not None:
            x, y = args.patch_pos
            eff_ps = args.patch_size
        else:
            x = int(H * args.min_row_ratio)
            eff_ps = compute_perspective_size(x, H, args.patch_size, args.perspective_min_scale)
            y_center = int((W - eff_ps) * args.patch_y_ratio)
            y = max(0, min(W - eff_ps, y_center))

        # Apply patch
        patched_t: torch.Tensor = apply_patch(img_t, patch, eff_ps, (x, y))

        # Convert to BGR numpy for YOLO
        clean_bgr = tensor_to_bgr(img_t)
        patched_bgr = tensor_to_bgr(patched_t)

        # YOLO inference
        dets_clean = detect_persons(yolo, clean_bgr, args.conf)
        dets_attacked = detect_persons(yolo, patched_bgr, args.conf)

        n_clean = len(dets_clean)
        n_attacked = len(dets_attacked)
        total_clean += n_clean
        total_attacked += n_attacked

        if n_clean > 0:
            images_with_person += 1
            if n_attacked == 0:
                disappeared += 1
            max_conf_clean = max(d["conf"] for d in dets_clean)
            conf_clean_list.append(max_conf_clean)
            max_conf_attacked = max((d["conf"] for d in dets_attacked), default=0.0)
            conf_attacked_list.append(max_conf_attacked)

        # Visualization frame (letterbox preserves aspect ratio)
        disp_clean, lb_scale, lb_px, lb_py = letterbox(clean_bgr, sz)
        disp_attacked, _, _, _ = letterbox(patched_bgr, sz)

        def scale_dets(dets):
            out = []
            for d in dets:
                x1, y1, x2, y2 = d["bbox"]
                out.append({"bbox": (int(x1 * lb_scale) + lb_px,
                                     int(y1 * lb_scale) + lb_py,
                                     int(x2 * lb_scale) + lb_px,
                                     int(y2 * lb_scale) + lb_py),
                             "conf": d["conf"]})
            return out

        disp_clean = draw_detections(disp_clean, scale_dets(dets_clean),
                                     YOLO_COLORS["clean"], "")
        disp_attacked = draw_detections(disp_attacked, scale_dets(dets_attacked),
                                        YOLO_COLORS["attacked"], "")

        # Patch thumbnail
        patch_vis = (patch.detach().cpu().permute(1, 2, 0).numpy().clip(0, 1) * 255).astype(np.uint8)
        patch_vis = cv2.resize(patch_vis, (sz, sz), interpolation=cv2.INTER_LINEAR)
        patch_vis = cv2.cvtColor(patch_vis, cv2.COLOR_RGB2BGR)
        cv2.putText(patch_vis, "Patch", (5, sz - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Running metrics for summary panel
        n_imgs_so_far = max(1, images_with_person)
        metrics_live = {
            "n_images": len(conf_clean_list),
            "source_class": args.source_class,
            "avg_det_clean": total_clean / max(1, len(paths)),
            "avg_det_attacked": total_attacked / max(1, len(paths)),
            "disappearance_rate": disappeared / n_imgs_so_far,
            "avg_conf_clean": float(np.mean(conf_clean_list)) if conf_clean_list else 0.0,
            "avg_conf_attacked": float(np.mean(conf_attacked_list)) if conf_attacked_list else 0.0,
            "avg_conf_drop": (float(np.mean(conf_clean_list)) - float(np.mean(conf_attacked_list)))
                             if conf_clean_list else 0.0,
        }
        summary = make_summary_panel(metrics_live, sz)

        # Labels
        for panel, label in [(disp_clean, f"Clean  ({n_clean} persons)"),
                              (disp_attacked, f"Attacked ({n_attacked} persons)")]:
            cv2.putText(panel, label, (5, sz - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        frame = np.hstack([disp_clean, disp_attacked, patch_vis, summary])

        if writer is not None:
            writer.write(frame)
        else:
            cv2.imshow("Transferability Eval", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("Interrupted by user.")
                break

    if writer is not None:
        writer.release()
        print(f"Video saved: {args.output}")
    else:
        cv2.destroyAllWindows()

    # Final report
    n = max(1, images_with_person)
    avg_conf_clean = float(np.mean(conf_clean_list)) if conf_clean_list else 0.0
    avg_conf_attacked = float(np.mean(conf_attacked_list)) if conf_attacked_list else 0.0

    print("\n" + "=" * 50)
    print("TRANSFERABILITY RESULTS (DINOv3 patch -> YOLO)")
    print("=" * 50)
    print(f"Dataset         : {args.dataset}")
    print(f"Patch           : {args.patch}")
    print(f"YOLO model      : {args.yolo_model}")
    print(f"Images evaluated: {len(paths)}")
    print(f"Images w/ person: {images_with_person}")
    print()
    print(f"Avg detections (clean)   : {total_clean / len(paths):.2f}/img")
    print(f"Avg detections (attacked): {total_attacked / len(paths):.2f}/img")
    print(f"Detection drop           : {(total_clean - total_attacked) / max(1, total_clean):.1%}")
    print()
    print(f"Disappearance rate       : {disappeared}/{images_with_person} = {disappeared/n:.1%}")
    print(f"  (images where all detections vanished)")
    print()
    print(f"Avg max confidence (clean)   : {avg_conf_clean:.4f}")
    print(f"Avg max confidence (attacked): {avg_conf_attacked:.4f}")
    print(f"Avg confidence drop          : {avg_conf_clean - avg_conf_attacked:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
