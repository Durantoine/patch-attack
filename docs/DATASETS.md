# Datasets pour Patch Attack

## Datasets recommandés pour tester les attaques adverses

### 1. **CIFAR-10** (Recommandé pour débuter)
- **Taille**: 60,000 images (50k train, 10k test)
- **Classes**: 10 classes (avion, auto, oiseau, chat, etc.)
- **Résolution**: 32×32 (upscalé à 224×224)
- **Avantages**:
  - Petit, rapide à télécharger et entraîner
  - Idéal pour prototyper
  - Inclus dans torchvision
- **Usage**:
  ```python
  from data.dataset import load_cifar10
  train_loader, val_loader, num_classes = load_cifar10()
  ```

### 2. **CIFAR-100**
- **Taille**: 60,000 images
- **Classes**: 100 classes
- **Résolution**: 32×32
- **Avantages**: Plus de classes que CIFAR-10, plus challengeant

### 3. **ImageNet** (Standard pour benchmarks)
- **Taille**: ~1.3M images train, 50k validation
- **Classes**: 1000 classes
- **Résolution**: Variable (224×224 après resize)
- **Avantages**:
  - Standard de l'industrie
  - DINOv3 pré-entraîné dessus
  - Meilleur pour évaluer la robustesse
- **Inconvénients**:
  - ~150GB, long à télécharger
  - Nécessite registration
- **Download**: https://image-net.org/download.php

### 4. **Tiny ImageNet**
- **Taille**: 100,000 images
- **Classes**: 200 classes
- **Résolution**: 64×64
- **Avantages**: Version réduite d'ImageNet, plus rapide
- **Download**: http://cs231n.stanford.edu/tiny-imagenet-200.zip

### 5. **GTSRB** (German Traffic Sign Recognition)
- **Taille**: ~50,000 images
- **Classes**: 43 classes (panneaux de signalisation)
- **Avantages**:
  - Très pertinent pour patch attacks (security critical)
  - Real-world application
  - Images variées en taille et conditions
- **Download**: https://benchmark.ini.rub.de/gtsrb_dataset.html

### 6. **STL-10**
- **Taille**: 13,000 images
- **Classes**: 10 classes
- **Résolution**: 96×96
- **Avantages**: Images plus grandes que CIFAR, dataset compact

## Recommandation selon votre objectif

### Pour débuter rapidement
→ **CIFAR-10** - Le setup est déjà prêt dans le code

### Pour un projet académique sérieux
→ **Tiny ImageNet** ou **GTSRB** - Bon compromis taille/pertinence

### Pour publication / benchmark
→ **ImageNet** - Standard reconnu

### Pour application pratique
→ **GTSRB** - Très pertinent pour démontrer les risques de sécurité

## Installation rapide

```bash
# CIFAR-10 (automatique via torchvision)
python src/train.py  # Télécharge automatiquement

# ImageNet (manuel)
# 1. S'inscrire sur image-net.org
# 2. Télécharger train/val
# 3. Extraire dans data/raw/imagenet/

# Tiny ImageNet
cd data/raw
wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
unzip tiny-imagenet-200.zip
```

## Structure de dossier attendue

```
data/raw/
├── cifar-10-batches-py/     # Auto-téléchargé
├── imagenet/                 # Manuel
│   ├── train/
│   └── val/
└── tiny-imagenet-200/        # Manuel
    ├── train/
    └── val/
```
