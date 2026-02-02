import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.dinov3_loader import load_dinov3
from src.segmentation.unsupervised import DINOv3Segmenter, detect_boundaries
from src.attacks.segmentation_attack import SegmentationCoherenceAttack

DINOV3_WEIGHTS_PATH = "src/models/weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"


def denormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """Denormalize image tensor for visualization."""
    mean = torch.tensor(mean).view(3, 1, 1).to(tensor.device)
    std = torch.tensor(std).view(3, 1, 1).to(tensor.device)
    return tensor * std + mean


def overlay_boundaries(image_np, boundaries, color=[1, 0, 0], thickness=1):
    """Overlay boundary pixels on image."""
    overlay = image_np.copy()
    boundary_coords = np.argwhere(boundaries)
    for y, x in boundary_coords:
        y_img = int(y * image_np.shape[0] / boundaries.shape[0])
        x_img = int(x * image_np.shape[1] / boundaries.shape[1])
        for dy in range(-thickness, thickness+1):
            for dx in range(-thickness, thickness+1):
                yi, xi = y_img + dy, x_img + dx
                if 0 <= yi < overlay.shape[0] and 0 <= xi < overlay.shape[1]:
                    overlay[yi, xi] = color
    return overlay


def main():
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}\n")

    checkpoint_path = DINOV3_WEIGHTS_PATH
    if not Path(checkpoint_path).exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return

    print("Loading DINOv3 model...")
    model = load_dinov3(checkpoint_path, device=device)
    print("Model loaded successfully\n")

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
    print(f"Image: {image_path}\n")

    print("Initializing unsupervised segmentation...")
    segmenter = DINOv3Segmenter(model, n_clusters=5, patch_size=16, img_size=224)

    print("Computing original segmentation...")
    seg_orig, patch_tokens_orig, _ = segmenter.segment(image)
    coherence_orig = segmenter.compute_segment_coherence(patch_tokens_orig, seg_orig)
    boundary_orig = segmenter.compute_boundary_length(seg_orig)

    print(f"Original segmentation:")
    print(f"  Coherence: {coherence_orig:.4f}")
    print(f"  Boundary length: {boundary_orig:.1f}\n")

    print("Initializing adversarial patch attack...")
    attack = SegmentationCoherenceAttack(model, segmenter, patch_size=50, device=device)

    print("Running attack (200 iterations)...")
    print("Objective: Destroy spatial coherence\n")

    results = attack.attack(
        image=image,
        num_iterations=200,
        learning_rate=0.05,
        attack_mode='coherence'
    )

    print("\n" + "="*60)
    print("ATTACK RESULTS")
    print("="*60)
    print(f"Coherence (original):     {results['coherence_original']:.4f}")
    print(f"Coherence (attacked):     {results['coherence_attacked']:.4f}")
    print(f"Coherence degradation:    {results['coherence_degradation']:.4f}")
    print(f"Boundary length (orig):   {results['boundary_original']:.1f}")
    print(f"Boundary length (att):    {results['boundary_attacked']:.1f}")
    print(f"Boundary increase:        {results['boundary_increase']:.1f}")
    print(f"Segmentation IoU:         {results['iou']:.4f}")
    print(f"Attack success:           {'Yes' if results['attack_success'] else 'No'}")
    print("="*60)

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

    img_np = denormalize(image[0]).cpu().permute(1, 2, 0).numpy()
    img_np = np.clip(img_np, 0, 1)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(img_np)
    ax1.set_title("Original Image", fontsize=12)
    ax1.axis('off')

    ax2 = fig.add_subplot(gs[0, 1])
    seg_rgb_orig = segmenter.visualize_segmentation(results['segmentation_original'])
    seg_rgb_orig_upscaled = np.kron(seg_rgb_orig, np.ones((16, 16, 1)))
    ax2.imshow(seg_rgb_orig_upscaled)
    ax2.set_title(f"Original Segmentation\n{segmenter.n_clusters} clusters", fontsize=12)
    ax2.axis('off')

    ax3 = fig.add_subplot(gs[0, 2])
    boundaries_orig = detect_boundaries(results['segmentation_original'])
    img_with_bounds_orig = overlay_boundaries(img_np, boundaries_orig, color=[1, 0, 0])
    ax3.imshow(img_with_bounds_orig)
    ax3.set_title(f"Original Boundaries\nLength: {results['boundary_original']:.0f}", fontsize=12)
    ax3.axis('off')

    ax4 = fig.add_subplot(gs[0, 3])
    patch_np = results['patch'][0].permute(1, 2, 0).numpy()
    patch_np = (patch_np - patch_np.min()) / (patch_np.max() - patch_np.min() + 1e-8)
    ax4.imshow(patch_np)
    ax4.set_title("Adversarial Patch (50x50)", fontsize=12)
    ax4.axis('off')

    with torch.no_grad():
        patched_img = attack.apply_patch(image, results['position'])

    patched_np = denormalize(patched_img[0]).cpu().permute(1, 2, 0).numpy()
    patched_np = np.clip(patched_np, 0, 1)

    ax5 = fig.add_subplot(gs[1, 0])
    ax5.imshow(patched_np)
    ax5.set_title("Image with Adversarial Patch", fontsize=12)
    ax5.axis('off')

    ax6 = fig.add_subplot(gs[1, 1])
    seg_rgb_att = segmenter.visualize_segmentation(results['segmentation_attacked'])
    seg_rgb_att_upscaled = np.kron(seg_rgb_att, np.ones((16, 16, 1)))
    ax6.imshow(seg_rgb_att_upscaled)
    ax6.set_title(f"Attacked Segmentation\nIoU: {results['iou']:.3f}", fontsize=12)
    ax6.axis('off')

    ax7 = fig.add_subplot(gs[1, 2])
    boundaries_att = detect_boundaries(results['segmentation_attacked'])
    img_with_bounds_att = overlay_boundaries(patched_np, boundaries_att, color=[1, 0, 0])
    ax7.imshow(img_with_bounds_att)
    ax7.set_title(f"Attacked Boundaries\nLength: {results['boundary_attacked']:.0f}", fontsize=12)
    ax7.axis('off')

    ax8 = fig.add_subplot(gs[1, 3])
    seg_diff = (results['segmentation_original'] != results['segmentation_attacked']).astype(float)
    seg_diff_upscaled = np.kron(seg_diff, np.ones((16, 16)))
    ax8.imshow(seg_diff_upscaled, cmap='Reds', interpolation='bilinear', vmin=0, vmax=1)
    changed_pixels = seg_diff.sum()
    total_pixels = seg_diff.size
    ax8.set_title(f"Segmentation Changes\n{changed_pixels:.0f}/{total_pixels} patches ({100*changed_pixels/total_pixels:.1f}%)", fontsize=10)
    ax8.axis('off')

    ax9 = fig.add_subplot(gs[2, :2])
    ax9_twin = ax9.twinx()
    line1 = ax9.plot(results['history']['iterations'], results['history']['coherences'],
                     'b-', linewidth=2, label='Coherence')
    line2 = ax9_twin.plot(results['history']['iterations'], results['history']['boundaries'],
                          'r-', linewidth=2, label='Boundary Length')
    ax9.set_xlabel('Iteration', fontsize=11)
    ax9.set_ylabel('Coherence', fontsize=11, color='b')
    ax9_twin.set_ylabel('Boundary Length', fontsize=11, color='r')
    ax9.set_title('Attack Progress', fontsize=12)
    ax9.grid(True, alpha=0.3)
    ax9.tick_params(axis='y', labelcolor='b')
    ax9_twin.tick_params(axis='y', labelcolor='r')
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax9.legend(lines, labels, loc='upper left', fontsize=10)

    ax10 = fig.add_subplot(gs[2, 2:])
    ax10.plot(results['history']['iterations'], results['history']['losses'],
              'g-', linewidth=2)
    ax10.set_xlabel('Iteration', fontsize=11)
    ax10.set_ylabel('Loss', fontsize=11)
    ax10.set_title('Loss Evolution', fontsize=12)
    ax10.grid(True, alpha=0.3)

    output_path = "results/segmentation_attack.png"
    Path("results").mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nResults saved to {output_path}")
    plt.show()


if __name__ == "__main__":
    main()
