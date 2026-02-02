# Adversarial Patch Attacks on DINOv3 Segmentation

This document outlines research ideas for adversarial patch attacks targeting segmentation based on DINOv3 foundation model embeddings.

## Idea 1: Unsupervised Segmentation + Spatial Coherence Attack

### Concept
- Perform unsupervised segmentation from DINOv3 patch tokens
- Add a local adversarial patch
- Objective: destroy spatial coherence of segments

### Pipeline
1. Image → DINOv3 patch tokens (N_patches, D)
2. Spatial clustering (k-means, spectral clustering, or lightweight CRF)
3. Obtain pseudo-segmentation
4. Optimize patch to:
   - Merge two distinct objects
   - Fragment a single object into multiple segments

### Loss Functions
- Maximize entropy of labels within a region
- Minimize inter-patch similarity inside a segment

### Research Question
Are DINOv3 segments globally robust to local perturbations?

---

## Idea 2: Patch Attack to Hijack Segmentation Boundaries

### Concept
- Segmentation is initially correct
- Small adversarial patch attracts/shifts the segmentation boundary

### Objectives
- Make the model believe the patch belongs to the object
- Make the model believe the object belongs to the background

### Metrics
- IoU before/after patch
- Average boundary displacement distance

### Advantages
Very visual, perfect for figures and demonstrations.

---

## Idea 3: Combined Patch Attack on SAM + DINOv3

### Concept
- DINOv3 provides embeddings
- SAM (Segment Anything) performs segmentation
- Patch modifies DINOv3 features → SAM segments incorrectly

### Scenario
1. Image → DINOv3 embeddings
2. Embeddings → SAM prompt/guidance
3. Patch attacks only the image input

### Contribution
Demonstrate that SAM inherits vulnerabilities from DINOv3.

### Relevance
Very current topic (foundation models + robustness)

---

## Idea 4: Patch Attack for Mask Theft (Targeted Segmentation Attack)

### Concept
- Choose a target mask (e.g., from another image)
- Optimize patch so that:
  - The model segments the wrong object
  - Segmentation matches the target mask

### Applications
- Fool robotic vision systems
- Attack autonomous perception

---

## Idea 5: Patch Attack to Hallucinate Objects

### Concept
- Small adversarial patch
- Model segments an object that doesn't exist

### Examples
- Fake traffic sign
- Fake face
- Fake salient object region

### Relevance
Very close to physical adversarial attacks in the real world.

---

## Idea 6: Patch Attack + Temporal Consistency (Video)

### Concept
- Static patch on an object
- Video → frame-by-frame segmentation
- Patch causes:
  - Temporal instability
  - Mask flickering across frames

### Objectives
- Break temporal consistency
- Cause segmentation to vary wildly between consecutive frames

### Metrics
- Temporal IoU variance
- Mask stability score
- Frame-to-frame boundary displacement

---

## Implementation Priority

### Phase 1: Basic Unsupervised Segmentation (Idea 1)
1. Extract DINOv3 dense features (patch tokens)
2. Implement spatial clustering (k-means on patch embeddings)
3. Visualize pseudo-segmentation
4. Implement patch attack to maximize segment entropy

### Phase 2: Boundary Attack (Idea 2)
1. Detect segmentation boundaries
2. Optimize patch to shift boundaries
3. Measure IoU degradation

### Phase 3: SAM Integration (Idea 3)
1. Load SAM model
2. Use DINOv3 embeddings as SAM guidance
3. Attack DINOv3 to fool SAM segmentation

### Phase 4: Advanced Attacks (Ideas 4-6)
- Targeted mask theft
- Object hallucination
- Temporal consistency attacks (requires video)

---

## Key Technical Components

### DINOv3 Dense Features
```python
# Extract patch tokens (not just CLS token)
features = model.forward_features(image)  # Returns all patch tokens
patch_tokens = features[:, 1:, :]  # Skip CLS token
# Shape: (B, N_patches, D) where N_patches = (H/16) * (W/16)
```

### Unsupervised Segmentation
```python
from sklearn.cluster import KMeans

# Reshape patch tokens to spatial grid
B, N, D = patch_tokens.shape
H_patches, W_patches = 14, 14  # For 224x224 image with patch_size=16
tokens_2d = patch_tokens.reshape(B, H_patches, W_patches, D)

# Cluster in feature space
tokens_flat = tokens_2d.reshape(-1, D)
kmeans = KMeans(n_clusters=5)
labels = kmeans.fit_predict(tokens_flat)
segmentation_map = labels.reshape(H_patches, W_patches)
```

### Spatial Coherence Loss
```python
# Maximize entropy within a region
def segment_entropy_loss(patch_tokens, segmentation_map, region_mask):
    # High entropy = more label diversity = broken coherence
    region_labels = segmentation_map[region_mask]
    probs = torch.bincount(region_labels) / len(region_labels)
    entropy = -(probs * torch.log(probs + 1e-8)).sum()
    return -entropy  # Maximize entropy → minimize negative entropy
```

### Boundary Shift Loss
```python
# Measure how much boundaries moved
def boundary_shift_loss(seg_orig, seg_attacked):
    boundaries_orig = detect_boundaries(seg_orig)
    boundaries_attacked = detect_boundaries(seg_attacked)
    shift_distance = chamfer_distance(boundaries_orig, boundaries_attacked)
    return -shift_distance  # Maximize shift
```

---

## Datasets

### Recommended
- COCO-Stuff: Images with segmentation annotations
- PASCAL VOC: Object segmentation
- Custom: Simple objects on clean backgrounds (for clear visualization)

### For Initial Experiments
Use simple images with 2-3 clear objects to validate concepts.

---

## Metrics

### Segmentation Quality
- IoU (Intersection over Union)
- Boundary F1 score
- Segment count stability

### Attack Success
- IoU degradation: `(IoU_orig - IoU_attacked) / IoU_orig`
- Boundary displacement (pixels)
- Segment fragmentation: number of connected components
- Temporal variance (for video)

---

## Visualization

### Key Figures to Generate
1. Original image + pseudo-segmentation
2. Image with adversarial patch + attacked segmentation
3. Difference map showing segment changes
4. Boundary overlay (before/after)
5. Evolution curves: IoU, entropy, boundary shift over iterations

---

## Research Questions

1. How robust are DINOv3-based segmentations to local perturbations?
2. Can a small patch (5-10% of image) break global segmentation?
3. Is spatial coherence preserved under adversarial attacks?
4. Do foundation models (SAM + DINOv3) amplify vulnerabilities?
5. Can patches transfer across different clustering methods?
6. Are attacks more effective on object boundaries vs. interiors?

---

## Next Steps

1. Implement dense feature extraction from DINOv3
2. Add k-means clustering for unsupervised segmentation
3. Create visualization pipeline
4. Implement Idea 1: spatial coherence attack
5. Measure attack success with IoU and entropy metrics
6. Generate figures for paper/presentation
