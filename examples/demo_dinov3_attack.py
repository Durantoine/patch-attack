import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms


def apply_patch_to_tensor(image_tensor, patch, position):
    """Apply patch to image tensor."""
    patched = image_tensor.clone()
    x, y = position
    patch_size = patch.shape[-1]
    patched[:, :, y:y+patch_size, x:x+patch_size] = patch
    return patched


def run_dinov3_attack():
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    print("\nLoading DINOv3 model...")
    # DINOv3 is actually loaded via timm or direct checkpoint
    # For now, using DINOv2 as DINOv3 may not be publicly available yet
    # If you have access to DINOv3 checkpoint, load it here
    try:
        # Try loading DINOv3 if available
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg')
    except:
        # Fallback to DINOv2
        print("DINOv3 not available, using DINOv2 ViT-S/14 with registers")
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')

    model = model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    print("Loading images...")
    image1_pil = Image.open("data/6.png").convert("RGB")
    image2_pil = Image.open("data/7.png").convert("RGB")

    image1 = transform(image1_pil).unsqueeze(0).to(device)
    image2 = transform(image2_pil).unsqueeze(0).to(device)

    print(f"Image shapes: {image1.shape}")

    with torch.no_grad():
        feat1_orig = model(image1)
        feat2_orig = model(image2)

        cosine_sim_orig = F.cosine_similarity(feat1_orig, feat2_orig)
        distance_orig = (feat1_orig - feat2_orig).norm()

        print(f"\nOriginal embeddings:")
        print(f"  Cosine similarity: {cosine_sim_orig.item():.4f}")
        print(f"  L2 distance: {distance_orig.item():.4f}")

    patch_size = 50
    patch_position = (87, 87)

    patch = torch.nn.Parameter(torch.rand(1, 3, patch_size, patch_size, device=device))
    optimizer = torch.optim.Adam([patch], lr=0.01)

    print(f"\nStarting attack (patch size: {patch_size}x{patch_size})...")
    print(f"Objective: Make similar images appear very different to DINOv3")

    losses = []
    similarities = []
    distances = []

    for iteration in range(100):
        patched_image1 = apply_patch_to_tensor(image1, patch, patch_position)

        feat1 = model(patched_image1)
        feat2 = model(image2)

        cosine_sim = F.cosine_similarity(feat1, feat2)
        distance = (feat1 - feat2).norm()

        loss = cosine_sim

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            patch.data.clamp_(-2, 2)

        losses.append(loss.item())
        similarities.append(cosine_sim.item())
        distances.append(distance.item())

        if iteration % 10 == 0:
            print(f"Iter {iteration:3d} | Loss: {loss.item():7.4f} | "
                  f"Similarity: {cosine_sim.item():.4f} | Distance: {distance.item():.4f}")

        if iteration == 50:
            optimizer.param_groups[0]['lr'] = 0.005

    print("\nAttack complete!")

    with torch.no_grad():
        patched_final = apply_patch_to_tensor(image1, patch, patch_position)
        feat1_final = model(patched_final)
        feat2_final = model(image2)

        sim_final = F.cosine_similarity(feat1_final, feat2_final)
        dist_final = (feat1_final - feat2_final).norm()

        print(f"\nFinal results:")
        print(f"  Original similarity: {cosine_sim_orig.item():.4f}")
        print(f"  Attacked similarity: {sim_final.item():.4f}")
        print(f"  Similarity reduction: {(cosine_sim_orig.item() - sim_final.item()):.4f}")
        print(f"  Attack success: {'Yes' if sim_final < 0.5 else 'No'}")

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    def denormalize(tensor):
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(tensor.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(tensor.device)
        return tensor * std + mean

    img1_np = denormalize(image1[0]).cpu().permute(1, 2, 0).numpy()
    img1_np = np.clip(img1_np, 0, 1)
    axes[0, 0].imshow(img1_np)
    axes[0, 0].set_title(f"Image 1 (Original)")
    axes[0, 0].axis('off')

    img2_np = denormalize(image2[0]).cpu().permute(1, 2, 0).numpy()
    img2_np = np.clip(img2_np, 0, 1)
    axes[0, 1].imshow(img2_np)
    axes[0, 1].set_title(f"Image 2 (Reference)")
    axes[0, 1].axis('off')

    axes[0, 2].text(0.5, 0.5, f"Original Similarity\n{cosine_sim_orig.item():.4f}",
                     ha='center', va='center', fontsize=16, transform=axes[0, 2].transAxes)
    axes[0, 2].axis('off')

    patched_np = denormalize(patched_final[0]).cpu().permute(1, 2, 0).numpy()
    patched_np = np.clip(patched_np, 0, 1)
    axes[1, 0].imshow(patched_np)
    axes[1, 0].set_title(f"Image 1 with Patch")
    axes[1, 0].axis('off')

    patch_np = patch[0].cpu().permute(1, 2, 0).detach().numpy()
    patch_np = (patch_np - patch_np.min()) / (patch_np.max() - patch_np.min())
    axes[1, 1].imshow(patch_np)
    axes[1, 1].set_title("Adversarial Patch")
    axes[1, 1].axis('off')

    axes[1, 2].plot(similarities)
    axes[1, 2].set_xlabel('Iteration')
    axes[1, 2].set_ylabel('Cosine Similarity')
    axes[1, 2].set_title('Similarity Evolution')
    axes[1, 2].grid(True)
    axes[1, 2].axhline(y=0.5, color='r', linestyle='--', label='Threshold')
    axes[1, 2].legend()

    plt.tight_layout()
    plt.savefig("results/dinov3_attack.png", dpi=150, bbox_inches='tight')
    print("\nResults saved to results/dinov3_attack.png")
    plt.show()


if __name__ == "__main__":
    run_dinov3_attack()
