import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.dinov3_loader import load_dinov3
from src.attacks.dinov3_attack import DINOv3PatchAttack

DINOV3_WEIGHTS_PATH = "src/models/weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"

def denormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """Denormalize image tensor for visualization."""
    mean = torch.tensor(mean).view(3, 1, 1).to(tensor.device)
    std = torch.tensor(std).view(3, 1, 1).to(tensor.device)
    return tensor * std + mean


def main():
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}\n")

    checkpoint_path = DINOV3_WEIGHTS_PATH
    if not Path(checkpoint_path).exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return

    print("\nLoading DINOv3 model...")
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

    print("Initializing adversarial patch attack...")
    attack = DINOv3PatchAttack(model, patch_size=50, device=device)

    print("Running attack (100 iterations)...")
    print("Objective: Maximize L2 distance from original embedding\n")

    results = attack.attack_embedding(
        image=image,
        num_iterations=100,
        learning_rate=0.01,
        attack_type='maximize_distance'
    )

    print("="*60)
    print("ATTACK RESULTS")
    print("="*60)
    print(f"Original embedding norm:  {results['embedding_norm_original']:.4f}")
    print(f"Attacked embedding norm:  {results['embedding_norm_attacked']:.4f}")
    print(f"L2 distance (orig->att):  {results['l2_distance']:.4f}")
    print(f"Cosine similarity:        {results['cosine_similarity']:.4f}")
    print(f"Attack success:           {'Yes' if results['attack_success'] else 'No'}")
    print("="*60)

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    img_np = denormalize(image[0]).cpu().permute(1, 2, 0).numpy()
    img_np = np.clip(img_np, 0, 1)
    axes[0, 0].imshow(img_np)
    axes[0, 0].set_title(f"Original Image", fontsize=14)
    axes[0, 0].axis('off')

    with torch.no_grad():
        patched_img = attack.apply_patch(image, results['position'])

    patched_np = denormalize(patched_img[0]).cpu().permute(1, 2, 0).numpy()
    patched_np = np.clip(patched_np, 0, 1)
    axes[0, 1].imshow(patched_np)
    axes[0, 1].set_title(f"Image with Adversarial Patch", fontsize=14)
    axes[0, 1].axis('off')

    patch_np = results['patch'][0].permute(1, 2, 0).numpy()
    patch_np = (patch_np - patch_np.min()) / (patch_np.max() - patch_np.min() + 1e-8)
    axes[1, 0].imshow(patch_np)
    axes[1, 0].set_title("Adversarial Patch (50x50)", fontsize=14)
    axes[1, 0].axis('off')

    ax = axes[1, 1]
    ax2 = ax.twinx()

    line1 = ax.plot(results['history']['iterations'], results['history']['distances'], 'b-', linewidth=2, label='L2 Distance')
    line2 = ax2.plot(results['history']['iterations'], results['history']['cosine_similarities'], 'r-', linewidth=2, label='Cosine Similarity')

    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('L2 Distance', fontsize=12, color='b')
    ax2.set_ylabel('Cosine Similarity', fontsize=12, color='r')
    ax.set_title('Attack Progress', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='y', labelcolor='b')
    ax2.tick_params(axis='y', labelcolor='r')

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, fontsize=10, loc='upper left')

    plt.tight_layout()
    output_path = "results/dinov3_attack.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nResults saved to {output_path}")
    plt.show()


if __name__ == "__main__":
    main()
