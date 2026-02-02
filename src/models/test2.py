import torch
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import SpectralClustering
from torchvision import transforms
from PIL import Image
from pathlib import Path

MODEL_NAME = "dinov3_vits16"
DINOV3_LOCATION = str(Path(__file__).parent / "dinov3")

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

model = torch.hub.load(
    DINOV3_LOCATION,
    MODEL_NAME,
    source="local",
    pretrained=False
)

weights_path = Path(__file__).parent / "weights" / "dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
checkpoint = torch.load(str(weights_path), map_location=device, weights_only=False)
model.load_state_dict(checkpoint, strict=True)

model.eval().to(device)
for p in model.parameters():
    p.requires_grad = False

PATCH_SIZE = model.patch_size


transform = transforms.Compose([
    transforms.Resize(256),  # Resize shortest side to 256
    transforms.CenterCrop(224),  # Then crop to 224x224
    transforms.ToTensor(),
])

image_path = Path(__file__).parent.parent.parent / "data" / "stuttgart" / "stuttgart_000000_000019_leftImg8bit.png"
image = Image.open(str(image_path)).convert("RGB")
image = transform(image).to(device)



def get_patch_tokens(model, image):
    with torch.no_grad():
        features = model.get_intermediate_layers(image.unsqueeze(0), n=1)[0]
    tokens = features[0, 1:]  # remove CLS, shape: (N, D)
    tokens = F.normalize(tokens, dim=1)
    return tokens


def get_patch_tokens_diff(model, image):
    """Extract patch tokens with gradients enabled for backprop."""
    features = model.get_intermediate_layers(image.unsqueeze(0), n=1)[0]
    tokens = features[0, 1:]  # remove CLS, shape: (N, D)
    tokens = F.normalize(tokens, dim=1)
    return tokens


def patch_coordinates(H, W, P, device):
    h = H // P
    w = W // P
    ys, xs = torch.meshgrid(
        torch.arange(h, device=device),
        torch.arange(w, device=device),
        indexing="ij"
    )
    coords = torch.stack([ys, xs], dim=-1).reshape(-1, 2).float()
    coords /= max(h, w)
    return coords



def spectral_segmentation(tokens, coords, n_clusters=3):
    # Use only semantic tokens (stronger weight) + weak spatial
    X = torch.cat([tokens, 0.1 * coords], dim=1)
    X_np = X.cpu().numpy()

    spectral = SpectralClustering(
        n_clusters=n_clusters,
        affinity="nearest_neighbors",
        n_neighbors=15,  # More neighbors for smoother regions
        assign_labels="kmeans",
        random_state=0
    )

    labels = spectral.fit_predict(X_np)
    return labels




def apply_patch(image, patch, pos):
    x, y = pos
    patched = image.clone()
    patched[:, x:x+patch.shape[1], y:y+patch.shape[2]] = patch
    return patched


# =========================
# 6. Initialisation attaque
# =========================

patch_size = 32
patch_pos = (0, 0)
steps = 1500

# Option 1: Charger un patch pré-entraîné
LOAD_PATCH = True  # Mettre à False pour entraîner un nouveau patch
PATCH_PATH = Path(__file__).parent.parent.parent / "results" / "universal_patch_final.pt"

if LOAD_PATCH and PATCH_PATH.exists():
    print(f"Loading pre-trained patch from {PATCH_PATH}")
    patch = torch.load(str(PATCH_PATH), map_location=device, weights_only=True)
    patch = patch.to(device)
    # Redimensionner si nécessaire
    if patch.shape[1] != patch_size or patch.shape[2] != patch_size:
        patch = F.interpolate(patch.unsqueeze(0), size=(patch_size, patch_size), mode='bilinear', align_corners=False).squeeze(0)
    patch.requires_grad = False
    print(f"Patch loaded: shape {patch.shape}")
else:
    if LOAD_PATCH:
        print(f"Patch not found at {PATCH_PATH}, training new patch...")
    patch = torch.rand(
        3, patch_size, patch_size,
        device=device,
        requires_grad=True
    )

tokens_ref = get_patch_tokens(model, image)


# =========================
# 7. Loss d'attaque
# =========================

def attack_loss(tokens_ref, tokens_adv):
    # éloigner les représentations (maximize distance)
    return -F.mse_loss(tokens_adv, tokens_ref)


# =========================
# 8. Boucle d'optimisation (skip si patch chargé)
# =========================

if not LOAD_PATCH or not PATCH_PATH.exists():
    optimizer = torch.optim.Adam([patch], lr=0.05)

    for step in range(steps):
        optimizer.zero_grad()

        patched_img = apply_patch(image, patch, patch_pos)
        tokens_adv = get_patch_tokens_diff(model, patched_img)

        loss = attack_loss(tokens_ref, tokens_adv)
        loss.backward()
        optimizer.step()

        patch.data.clamp_(0, 1)

        if step % 20 == 0:
            mse = F.mse_loss(tokens_adv, tokens_ref).item()
            print(f"[{step:03d}] loss = {loss.item():.4f} | MSE = {mse:.6f}")
else:
    print("Skipping optimization (using pre-trained patch)")


# =========================
# 9. Segmentation finale
# =========================

# Get final adversarial tokens without gradients
patched_img_final = apply_patch(image, patch, patch_pos)
tokens_adv = get_patch_tokens(model, patched_img_final)

H, W = image.shape[1:]
coords = patch_coordinates(H, W, PATCH_SIZE, device)

# Trim coords to match actual token count
n_tokens = tokens_ref.shape[0]
coords = coords[:n_tokens]

labels_ref = spectral_segmentation(tokens_ref, coords)
labels_adv = spectral_segmentation(tokens_adv, coords)

# Pad labels to 196 for 14x14 reshape
h_patches = H // PATCH_SIZE
w_patches = W // PATCH_SIZE
expected_size = h_patches * w_patches

if len(labels_ref) < expected_size:
    pad_size = expected_size - len(labels_ref)
    labels_ref = np.concatenate([labels_ref, np.zeros(pad_size, dtype=labels_ref.dtype)])
    labels_adv = np.concatenate([labels_adv, np.zeros(pad_size, dtype=labels_adv.dtype)])

seg_ref = labels_ref.reshape(h_patches, w_patches)
seg_adv = labels_adv.reshape(h_patches, w_patches)

print("Segmentation terminée.")


# =========================
# 10. Visualisation
# =========================

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# PCA projection for visualization
tokens_ref_np = tokens_ref.cpu().numpy()
tokens_adv_np = tokens_adv.cpu().numpy()

# Combine tokens for PCA
all_tokens = np.vstack([tokens_ref_np, tokens_adv_np])
pca = PCA(n_components=2)
pca.fit(all_tokens)

# Project both
tokens_ref_2d = pca.transform(tokens_ref_np)
tokens_adv_2d = pca.transform(tokens_adv_np)

# Token distances
token_dist = np.linalg.norm(tokens_adv_np - tokens_ref_np, axis=1)

fig = plt.figure(figsize=(20, 10))
gs = fig.add_gridspec(2, 4, hspace=0.3, wspace=0.3)

# Row 1: Images and patch
ax1 = fig.add_subplot(gs[0, 0])
ax1.imshow(image.cpu().permute(1, 2, 0).numpy())
ax1.set_title("Original Image", fontsize=14, fontweight='bold')
ax1.axis('off')

ax2 = fig.add_subplot(gs[0, 1])
patched_display = patched_img_final.detach().cpu().permute(1, 2, 0).numpy()
ax2.imshow(np.clip(patched_display, 0, 1))
ax2.set_title(f"Adversarial Patch Attack\n{patch_size}x{patch_size} @ ({patch_pos[0]}, {patch_pos[1]})", fontsize=14, fontweight='bold')
ax2.axis('off')

ax3 = fig.add_subplot(gs[0, 2])
# PCA visualization
scatter1 = ax3.scatter(tokens_ref_2d[:, 0], tokens_ref_2d[:, 1],
                      c='blue', alpha=0.6, s=30, label='Original', edgecolors='none')
scatter2 = ax3.scatter(tokens_adv_2d[:, 0], tokens_adv_2d[:, 1],
                      c='red', alpha=0.6, s=30, label='Adversarial', edgecolors='none')
# Draw arrows for displacement
for i in range(0, len(tokens_ref_2d), 10):
    ax3.arrow(tokens_ref_2d[i, 0], tokens_ref_2d[i, 1],
             tokens_adv_2d[i, 0] - tokens_ref_2d[i, 0],
             tokens_adv_2d[i, 1] - tokens_ref_2d[i, 1],
             color='gray', alpha=0.3, width=0.001, head_width=0.02)
ax3.set_xlabel('PC1', fontsize=12)
ax3.set_ylabel('PC2', fontsize=12)
ax3.set_title(f'Token Embedding Space (PCA)\nMSE: {F.mse_loss(tokens_adv, tokens_ref).item():.6f}',
              fontsize=14, fontweight='bold')
ax3.legend(fontsize=11)
ax3.grid(True, alpha=0.3)

ax4 = fig.add_subplot(gs[0, 3])
# Distance histogram
ax4.hist(token_dist, bins=30, color='purple', alpha=0.7, edgecolor='black')
ax4.axvline(token_dist.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {token_dist.mean():.4f}')
ax4.set_xlabel('L2 Distance', fontsize=12)
ax4.set_ylabel('Number of Tokens', fontsize=12)
ax4.set_title(f'Token Displacement Distribution\nMax: {token_dist.max():.4f}',
              fontsize=14, fontweight='bold')
ax4.legend(fontsize=11)
ax4.grid(True, alpha=0.3, axis='y')

# Row 2: Segmentations
ax5 = fig.add_subplot(gs[1, 0])
seg_ref_upscaled = np.kron(seg_ref, np.ones((16, 16)))
ax5.imshow(seg_ref_upscaled, cmap='tab10', interpolation='nearest')
ax5.set_title("Original Segmentation", fontsize=14, fontweight='bold')
ax5.axis('off')

ax6 = fig.add_subplot(gs[1, 1])
seg_adv_upscaled = np.kron(seg_adv, np.ones((16, 16)))
ax6.imshow(seg_adv_upscaled, cmap='tab10', interpolation='nearest')
ax6.set_title("Adversarial Segmentation", fontsize=14, fontweight='bold')
ax6.axis('off')

ax7 = fig.add_subplot(gs[1, 2])
# Compute IoU with best label permutation
from itertools import permutations
seg1_flat = seg_ref.flatten()
seg2_flat = seg_adv.flatten()
n_labels = max(seg_ref.max(), seg_adv.max()) + 1

best_acc = 0.0
best_perm = None
for perm in permutations(range(n_labels)):
    seg2_permuted = np.array([perm[label] if label < len(perm) else label for label in seg2_flat])
    acc = (seg1_flat == seg2_permuted).sum() / len(seg1_flat)
    if acc > best_acc:
        best_acc = acc
        best_perm = perm

# Apply best permutation
seg_adv_aligned = np.array([best_perm[label] if label < len(best_perm) else label for label in seg_adv.flatten()])
seg_adv_aligned = seg_adv_aligned.reshape(seg_ref.shape)

# Now compute real difference
seg_diff = (seg_ref != seg_adv_aligned).astype(float)
seg_diff_upscaled = np.kron(seg_diff, np.ones((16, 16)))
ax7.imshow(seg_diff_upscaled, cmap='Reds', interpolation='bilinear', vmin=0, vmax=1)
changed_patches = seg_diff.sum()
total_patches = seg_diff.size
iou = best_acc
ax7.set_title(f"Segmentation Changes\n{changed_patches:.0f}/{total_patches} patches ({100*changed_patches/total_patches:.1f}%)\nIoU: {iou:.3f}", fontsize=14, fontweight='bold')
ax7.axis('off')

ax8 = fig.add_subplot(gs[1, 3])
# Token distance heatmap
token_dist_2d = token_dist.copy()
if len(token_dist_2d) < h_patches * w_patches:
    pad_size = h_patches * w_patches - len(token_dist_2d)
    token_dist_2d = np.concatenate([token_dist_2d, np.zeros(pad_size)])
token_dist_2d = token_dist_2d.reshape(h_patches, w_patches)
token_dist_upscaled = np.kron(token_dist_2d, np.ones((16, 16)))
im = ax8.imshow(token_dist_upscaled, cmap='hot', interpolation='bilinear')
plt.colorbar(im, ax=ax8, fraction=0.046)
ax8.set_title(f"Token Distance (L2)\nAvg: {token_dist.mean():.4f}", fontsize=14, fontweight='bold')
ax8.axis('off')

plt.tight_layout()
plt.savefig(Path(__file__).parent.parent.parent / "results" / "test2_analysis.png", dpi=150, bbox_inches='tight')
print(f"\nVisualization saved to results/test2_analysis.png")
print(f"\nAttack Statistics:")
print(f"  MSE: {F.mse_loss(tokens_adv, tokens_ref).item():.6f}")
print(f"  Avg L2 distance: {token_dist.mean():.4f}")
print(f"  Max L2 distance: {token_dist.max():.4f}")
print(f"  Cosine similarity: {F.cosine_similarity(tokens_adv, tokens_ref, dim=1).mean().item():.4f}")
print(f"  Changed patches: {changed_patches:.0f}/{total_patches} ({100*changed_patches/total_patches:.1f}%)")
plt.show()