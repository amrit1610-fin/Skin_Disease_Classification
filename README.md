# 🔬 Skin Disease Classification — HAM10000

A production-grade, modular deep learning pipeline for skin lesion classification
on the [HAM10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000) dataset.
Implements **EfficientNet-B0**, **ResNet34**, **MobileNetV3-Large**, a
**ResNet34 + XGBoost** hybrid, and a **soft-voting ensemble** — all backed by
a single unified data module.

---

## 📁 Project Structure

```
skin-disease/
├── src/
│   ├── __init__.py
│   ├── prepare_data.py          # Data preprocessing & stratified splits
│   ├── dataset.py               # HAM10000Dataset + Albumentations transforms
│   ├── train_efficientnet.py    # EfficientNet-B0 training pipeline
│   ├── train_resnet.py          # ResNet34 training pipeline
│   ├── train_mobilenet.py       # MobileNetV3-Large training pipeline
│   ├── train_resnet_xgboost.py  # ResNet34 encoder + XGBoost hybrid
│   └── ensemble_predict.py      # Soft-voting ensemble inference
├── notebooks/
│   └── skin-disease-classification-testing.ipynb   # Clean evaluation notebook
├── data_splits/                 # Auto-generated CSV index files (git-ignored)
│   ├── train_split.csv
│   ├── val_split.csv
│   └── test_split.csv
├── checkpoints/                 # Saved model weights (git-ignored)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🧬 Dataset

| Property        | Detail |
|-----------------|--------|
| Name            | HAM10000 (Human Against Machine with 10000 training images) |
| Source          | [Kaggle – kmader/skin-cancer-mnist-ham10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000) |
| Classes         | 7 (nv, mel, bkl, bcc, akiec, vasc, df) |
| Total images    | ~10,015 |
| Image format    | JPEG |

### Class Labels

| Code   | Full Name                              | Label |
|--------|----------------------------------------|-------|
| nv     | Melanocytic nevi                       | 0     |
| mel    | Melanoma                               | 1     |
| bkl    | Benign keratosis-like lesions          | 2     |
| bcc    | Basal cell carcinoma                   | 3     |
| akiec  | Actinic keratoses / Intraepithelial    | 4     |
| vasc   | Vascular lesions                       | 5     |
| df     | Dermatofibroma                         | 6     |

---

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/your-username/skin-disease.git
cd skin-disease

# Create and activate virtual environment (recommended)
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux / macOS

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 End-to-End Execution

### Step 1 — Prepare Data Splits

```bash
python src/prepare_data.py --root /kaggle/input/datasets/kmader
```

> **What it does:**
> - Auto-detects `HAM10000_metadata.csv` under the provided root
> - Matches image IDs to file paths across `ham10000_images_part_1` and `ham10000_images_part_2`
> - Performs a **stratified 70 / 15 / 15** train/val/test split (`random_state=42`)
> - Applies `RandomOverSampler(random_state=42)` **only** on the training fold
> - Saves `data_splits/train_split.csv`, `val_split.csv`, `test_split.csv`

**Custom path (non-Kaggle):**
```bash
python src/prepare_data.py --root /path/to/your/ham10000/dataset
```

---

### Step 2 — Train EfficientNet-B0

```bash
python src/train_efficientnet.py \
    --epochs 25 \
    --batch_size 32 \
    --lr 1e-4
```

Saves best checkpoint to `checkpoints/efficientnet_best.pth`.

---

### Step 3 — Train ResNet34

```bash
python src/train_resnet.py \
    --epochs 25 \
    --batch_size 32 \
    --lr 1e-4
```

Saves best checkpoint to `checkpoints/resnet_best.pth`.

---

### Step 4 — Train MobileNetV3-Large

```bash
python src/train_mobilenet.py \
    --epochs 25 \
    --batch_size 32 \
    --lr 1e-4
```

Saves best checkpoint to `checkpoints/mobilenet_best.pth`.

---

### Step 5 — Train ResNet34 + XGBoost Hybrid

> ⚠️ Requires `checkpoints/resnet_best.pth` from Step 3.

```bash
python src/train_resnet_xgboost.py \
    --n_estimators 500 \
    --max_depth 6
```

- Freezes the ResNet34 backbone and extracts 512-dim embeddings
- Trains `XGBClassifier` on those features
- Saves the XGBoost model to `checkpoints/resnet_xgboost.json`

---

### Step 6 — Ensemble Inference

> ⚠️ Requires at least one trained DL checkpoint from steps 2–4.

```bash
python src/ensemble_predict.py
```

- Runs soft-voting over all available DL checkpoints
- Prints per-model accuracy + combined `classification_report`
- Saves a confusion matrix figure to `checkpoints/confusion_matrix.png`

---

### Step 7 — Evaluation Notebook

```bash
jupyter notebook notebooks/skin-disease-classification-testing.ipynb
```

The notebook provides:
- Per-model evaluation with classification reports
- ResNet34 + XGBoost test evaluation
- Soft-voting ensemble evaluation
- Comparative accuracy / F1 summary table
- Confusion matrices (one per model)
- Per-class F1-score grouped bar chart

---

## 🏗️ Architecture

### Shared Data Module (`src/dataset.py`)

All models consume the same `HAM10000Dataset` class.

| Split    | Augmentations |
|----------|---------------|
| **Train** | Resize(224) → HFlip → VFlip → Rotate → ShiftScaleRotate → BrightnessContrast → HueSaturationValue → Normalize → ToTensor |
| **Val / Test** | Resize(224) → Normalize → ToTensor |

### Models

| Model | Backbone | Modified Layer | Params (approx.) |
|-------|----------|----------------|-------------------|
| EfficientNet-B0 | `efficientnet_b0` (ImageNet) | `classifier[1]` → Linear(1280, 7) | ~5.3M |
| ResNet34 | `resnet34` (ImageNet) | `fc` → Dropout + Linear(512, 7) | ~21.8M |
| MobileNetV3-Large | `mobilenet_v3_large` (ImageNet) | `classifier[3]` → Linear(1280, 7) | ~5.5M |
| ResNet34+XGBoost | ResNet34 backbone (frozen) + XGBClassifier | — | — |

### Training Configuration

| Setting | Value |
|---------|-------|
| Optimizer | AdamW |
| LR Scheduler | CosineAnnealingLR |
| Loss | CrossEntropyLoss with class weights |
| Image size | 224 × 224 |
| Default LR | 1e-4 |
| Default epochs | 25 |
| Default batch size | 32 |

---

## 📊 CLI Arguments Reference

All training scripts accept the following arguments:

| Argument | Default | Description |
|----------|---------|-------------|
| `--train_csv` | `data_splits/train_split.csv` | Path to train split CSV |
| `--val_csv` | `data_splits/val_split.csv` | Path to val split CSV |
| `--checkpoint` | `checkpoints/<model>_best.pth` | Where to save best weights |
| `--epochs` | `25` | Number of training epochs |
| `--batch_size` | `32` | Mini-batch size |
| `--lr` | `1e-4` | Initial learning rate |
| `--weight_decay` | `1e-4` | AdamW weight decay |
| `--num_workers` | `2` | DataLoader worker processes |
| `--seed` | `42` | Random seed |

---

## 🛠️ Requirements

See [`requirements.txt`](requirements.txt). Key dependencies:

- `torch >= 2.0.0`
- `torchvision >= 0.15.0`
- `albumentations == 1.4.11`
- `opencv-python-headless >= 4.8.0`
- `imbalanced-learn >= 0.11.0`
- `xgboost >= 2.0.0`
- `scikit-learn >= 1.3.0`

---

## 📜 License

This project is released under the **MIT License**.

---

## 🙏 Acknowledgements

- Dataset: [Philipp Tschandl et al. — HAM10000](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T)
- Kaggle dataset mirror: [kmader/skin-cancer-mnist-ham10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)
