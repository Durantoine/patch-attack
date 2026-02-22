# Adversarial Patch Attacks on DINOv3

Attaques par patch adversarial sur un classifieur sémantique construit au-dessus de DINOv3 (ViT-S/16). Le patch fait disparaître ou mal-classifier des piétons (ou toute autre classe) dans une séquence de conduite.

**Compatible Mac M3 (MPS) / CUDA / CPU**

---

## Pipeline

```
1. train_classifier.py   → entraîne une sonde linéaire sur les tokens DINOv3 (Cityscapes)
2. generate_patch.py  → optimise un patch adversarial contre le classifieur
3. visualize_sequence.py → visualise l'attaque sur une séquence + export vidéo
```

---

## Démarrage rapide

```bash
# Installer le projet (crée les commandes train-classifier, generate-patch, visualize-sequence)
uv sync

# Placer les poids DINOv3 :
# src/patch_attack/models/weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth

# 1. Entraîner le classifieur
python -m patch_attack.train_classifier

# 2. Générer le patch adversarial
python -m patch_attack.generate_patch

# 3. Visualiser sur une séquence
python -m patch_attack.visualize_sequence
```

Ou après `uv sync` : `train-classifier` · `generate-patch` · `visualize-sequence`

Tous les paramètres sont dans [`src/patch_attack/utils/config.py`](src/patch_attack/utils/config.py) — **aucun argument CLI**.

**Développement :**
```bash
uv sync --all-extras
inv check      # format + lint + typecheck + test
inv typecheck  # mypy uniquement
inv lint       # ruff uniquement
```

---

## Structure

```
.
├── data/
│   ├── stuttgart/                  # Séquences de conduite
│   │   ├── stuttgart_00/
│   │   ├── stuttgart_01/
│   │   └── stuttgart_02/           # Séquence par défaut (VIZ_DATASET)
│   ├── leftImg8bit_trainvaltest/   # Images Cityscapes
│   └── gtFine_trainvaltest/        # Labels Cityscapes
├── results/
│   ├── classifier.pt               # Classifieur entraîné
│   ├── targeted_patch_best.pt      # Meilleur patch (fooling rate max)
│   ├── targeted_patch_final.pt     # Patch final
│   ├── targeted_attack_results.png # Courbe fooling rate + patch
│   ├── patch_evolution/            # Snapshots du patch pendant l'entraînement
│   ├── patch_evolution.mp4         # Vidéo évolution du patch
│   └── demo/
│       ├── stuttgart_02.mp4        # Vidéo de l'attaque sur la séquence
│       └── stuttgart_02_analysis.png  # Graphe fooling rate 3 distances
└── src/
    └── patch_attack/
        ├── train_classifier.py
        ├── generate_patch.py
        ├── visualize_sequence.py
        ├── utils/
        │   ├── config.py           # Tous les paramètres
        │   └── viz.py              # Utilitaires de visualisation
        └── models/
            ├── dinov3/             # Code DINOv3
            ├── dinov3_loader.py
            └── weights/            # Poids (non tracés par git)
```

---

## Configuration

Tout se passe dans [`src/patch_attack/utils/config.py`](src/patch_attack/utils/config.py) :

```python
CITYSCAPES_IMAGES = "data/leftImg8bit_trainvaltest/leftImg8bit/train"
CITYSCAPES_LABELS = "data/gtFine_trainvaltest/gtFine/train"
DATASET           = "data/leftImg8bit_trainvaltest/leftImg8bit/train"
VIZ_DATASET       = "data/stuttgart/stuttgart_02"  # ou stuttgart_00, stuttgart_01
CLASSIFIER        = "results/classifier.pt"
PATCH             = "results/targeted_patch_best.pt"
OUTPUT_DIR        = "results"

IMG_SIZE = 672      # 224=14×14 | 448=28×28 | 672=42×42 tokens

CLF_EPOCHS = 20
CLF_LR     = 0.001

SOURCE_CLASS             = 11    # person (0=road, 13=car, …)
TARGET_CLASS             = -1    # -1 = non ciblé, sinon ID de classe cible
ATTACK_STEPS             = 3000
ATTACK_LR                = 0.05
ATTACK_BATCH_SIZE        = 4
ATTACK_MIN_SOURCE_TOKENS = 10    # images filtrées si < N tokens source
PATCH_SIZE               = 132   # taille sur l'image (~17 % de IMG_SIZE)
PATCH_RES                = 256   # résolution interne d'optimisation
PATCH_PERSPECTIVE_MIN_SCALE = 0.3  # 3× plus petit en haut (loin) qu'en bas (près)
```

---

## Étape 1 — Classifieur linéaire

Entraîne `nn.Linear(384, 19)` sur les embeddings de tokens DINOv3 avec les labels Cityscapes.

**Données requises** ([cityscapes-dataset.com](https://www.cityscapes-dataset.com/downloads/)) :
- `leftImg8bit_trainvaltest.zip` → `data/leftImg8bit_trainvaltest/leftImg8bit/train/`
- `gtFine_trainvaltest.zip` → `data/gtFine_trainvaltest/gtFine/train/`

Une fenêtre live montre GT vs prédictions sur 4 images représentatives à chaque époque.

**Sortie :** `results/classifier.pt`

---

## Étape 2 — Patch adversarial

Optimise un patch pour que les tokens de la classe source soient mal classifiés.

**Scaling perspective** : la taille du patch varie avec la position verticale — plus petit en haut (loin), plus grand en bas (près). La formule `compute_perspective_size(x)` assure une mise à l'échelle cohérente avec une caméra de conduite.

**Filtrage** : seules les images avec ≥ `ATTACK_MIN_SOURCE_TOKENS` tokens de la classe source sont utilisées, aussi bien au chargement qu'à chaque step.

**Contrainte de placement** : le patch est placé en dehors de la région de la classe source (20 essais max), pour éviter une attaque triviale par occultation.

**Visualisation live :**
```
[ Image+Patch | Seg Original | Seg Attaqué | Patch | Légende ]
```
Appuyer sur `q` pour arrêter prématurément.

**Sorties :**
- `results/targeted_patch_best.pt` — meilleur patch (fooling rate max)
- `results/targeted_patch_final.pt` — patch final
- `results/targeted_attack_results.png` — courbe + patch
- `results/patch_evolution.mp4` — vidéo d'évolution
- `results/patch_evolution_grid.png` — grille de snapshots

---

## Étape 3 — Visualisation séquence

Applique le patch sur chaque frame et affiche l'effet en mode **multi-distance** (loin / moyen / proche) avec scaling perspective correct à chaque distance.

**Layout :**
```
Row 1 : [ Seg Original | Loin | Moyen | Proche | Légende ]
Row 2 : [ PCA 3D scatter | Heatmap L2 Loin | Moyen | Proche ]
```

**Indicateurs par distance :**
- `FR: XX%` — fooling rate (% de tokens source mal classifiés)
- Cadre rouge + `DISPARU!` — tous les tokens source ont disparu

**Contrôles :** `q` quitter · `Espace` pause

**Sorties :**
- `results/demo/<dataset>.mp4` — vidéo de l'attaque
- `results/demo/<dataset>_analysis.png` — graphe fooling rate + frames de disparition

---

## Fonctionnement

```
Image (672×672)
     ↓ DINOv3 ViT-S/16
1764 tokens (384 dim) + 1 CLS
     ↓ nn.Linear(384 → 19)
Carte de segmentation (42×42)

Boucle d'optimisation :
  1. Échantillonner un batch d'images contenant la classe source
  2. Placer le patch (taille perspective-correcte, hors zone source)
  3. Extraire les tokens → logits classifieur
  4. Loss = -CE(logits[source], source_class)   # non ciblé
          ou CE(logits[source], target_class)   # ciblé
  5. Rétropropager → mettre à jour le patch
  6. Contraindre patch ∈ [0, 1]
```

---

## Classes Cityscapes

| ID | Classe | ID | Classe |
|----|--------|----|--------|
| 0 | road | 10 | sky |
| 1 | sidewalk | **11** | **person** |
| 2 | building | 12 | rider |
| 8 | vegetation | 13 | car |
| 9 | terrain | 18 | bicycle |
