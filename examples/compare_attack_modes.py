import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.dinov3_loader import load_dinov3
from src.segmentation.unsupervised import DINOv3Segmenter
from src.attacks.segmentation_attack import SegmentationCoherenceAttack

DINOV3_WEIGHTS_PATH = "src/models/weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"


def main():
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}\n")

    checkpoint_path = DINOV3_WEIGHTS_PATH
    if not Path(checkpoint_path).exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return

    print("Loading DINOv3 model...")
    model = load_dinov3(checkpoint_path, device=device)
    print("Model loaded\n")

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    print("Loading image...")
    image_path = "data/6.png"
    image_pil = Image.open(image_path).convert("RGB")
    image = transform(image_pil).unsqueeze(0).to(device)

    print("Initializing segmentation...")
    segmenter = DINOv3Segmenter(model, n_clusters=5, patch_size=16, img_size=224)

    print("Computing original segmentation...")
    seg_orig, _ = segmenter.segment(image)

    attack_modes = ['entropy', 'coherence', 'boundary']
    results_all = {}

    for mode in attack_modes:
        print(f"\n{'='*60}")
        print(f"Running {mode.upper()} attack...")
        print('='*60)

        attack = SegmentationCoherenceAttack(model, segmenter, patch_size=16, device=device)

        results = attack.attack(
            image=image,
            num_iterations=100,
            learning_rate=0.01,
            attack_mode=mode
        )

        results_all[mode] = results

        print(f"\nResults for {mode}:")
        print(f"  Coherence degradation: {results['coherence_degradation']:.4f}")
        print(f"  Boundary increase:     {results['boundary_increase']:.1f}")
        print(f"  IoU:                   {results['iou']:.4f}")
        print(f"  Attack success:        {results['attack_success']}")

    print(f"\n{'='*60}")
    print("COMPARISON SUMMARY")
    print('='*60)
    print(f"{'Mode':<12} {'Coherence Deg':<15} {'Boundary Inc':<15} {'IoU':<10} {'Success'}")
    print('-'*60)
    for mode in attack_modes:
        r = results_all[mode]
        print(f"{mode:<12} {r['coherence_degradation']:<15.4f} {r['boundary_increase']:<15.1f} {r['iou']:<10.4f} {str(r['attack_success'])}")

    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    fig.suptitle('Comparison of Attack Modes', fontsize=16, fontweight='bold')

    for idx, mode in enumerate(attack_modes):
        r = results_all[mode]

        axes[idx, 0].set_title(f'{mode.capitalize()}: Original', fontsize=10)
        axes[idx, 0].imshow(segmenter.visualize_segmentation(r['segmentation_original']))
        axes[idx, 0].axis('off')

        axes[idx, 1].set_title(f'{mode.capitalize()}: Attacked', fontsize=10)
        axes[idx, 1].imshow(segmenter.visualize_segmentation(r['segmentation_attacked']))
        axes[idx, 1].axis('off')

        axes[idx, 2].set_title(f'Coherence: {r["coherence_degradation"]:.3f}', fontsize=10)
        axes[idx, 2].plot(r['history']['iterations'], r['history']['coherences'], 'b-', linewidth=2)
        axes[idx, 2].axhline(r['coherence_original'], color='r', linestyle='--', label='Original')
        axes[idx, 2].set_xlabel('Iteration')
        axes[idx, 2].set_ylabel('Coherence')
        axes[idx, 2].grid(True, alpha=0.3)
        axes[idx, 2].legend()

        axes[idx, 3].set_title(f'Boundary: +{r["boundary_increase"]:.0f}', fontsize=10)
        axes[idx, 3].plot(r['history']['iterations'], r['history']['boundaries'], 'g-', linewidth=2)
        axes[idx, 3].axhline(r['boundary_original'], color='r', linestyle='--', label='Original')
        axes[idx, 3].set_xlabel('Iteration')
        axes[idx, 3].set_ylabel('Boundary Length')
        axes[idx, 3].grid(True, alpha=0.3)
        axes[idx, 3].legend()

    plt.tight_layout()
    output_path = "results/attack_modes_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nComparison saved to {output_path}")


if __name__ == "__main__":
    main()
