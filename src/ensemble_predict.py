"""
src/ensemble_predict.py
========================
Consensus Soft-Voting Ensemble Inference for HAM10000
------------------------------------------------------
- Loads saved checkpoints for EfficientNet-B0, ResNet34, MobileNetV3-Large
- Runs each model on the held-out test split (data_splits/test_split.csv)
- Applies softmax to get per-class probability distributions
- Averages the probability distributions (equal-weight soft voting)
- Prints per-model accuracy and a combined classification_report

Usage:
    python src/ensemble_predict.py \
        [--efficientnet_ckpt checkpoints/efficientnet_best.pth] \
        [--resnet_ckpt       checkpoints/resnet_best.pth]       \
        [--mobilenet_ckpt    checkpoints/mobilenet_best.pth]    \
        [--test_csv          data_splits/test_split.csv]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import HAM10000Dataset, get_transforms, NUM_CLASSES, CLASS_NAMES

# ─── Configuration ────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Soft-voting ensemble on HAM10000 test set")
    p.add_argument("--test_csv",          default="data_splits/test_split.csv")
    p.add_argument("--efficientnet_ckpt", default="checkpoints/efficientnet_best.pth")
    p.add_argument("--resnet_ckpt",       default="checkpoints/resnet_best.pth")
    p.add_argument("--mobilenet_ckpt",    default="checkpoints/mobilenet_best.pth")
    p.add_argument("--batch_size",        type=int, default=32)
    p.add_argument("--num_workers",       type=int, default=2)
    p.add_argument("--save_cm",           default="checkpoints/confusion_matrix.png",
                   help="Path to save the confusion matrix figure")
    return p.parse_args()


# ─── Model Loaders ────────────────────────────────────────────────────────────

def _load_efficientnet(ckpt_path: str, device: torch.device) -> nn.Module:
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, NUM_CLASSES),
    )
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    return model.to(device).eval()


def _load_resnet(ckpt_path: str, device: torch.device) -> nn.Module:
    model = models.resnet34(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, NUM_CLASSES),
    )
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    return model.to(device).eval()


def _load_mobilenet(ckpt_path: str, device: torch.device) -> nn.Module:
    model = models.mobilenet_v3_large(weights=None)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, NUM_CLASSES)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    return model.to(device).eval()


# ─── Inference ────────────────────────────────────────────────────────────────

@torch.no_grad()
def get_probabilities(model: nn.Module,
                      loader: DataLoader,
                      device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """
    Run inference and return (probabilities, true_labels).

    Returns
    -------
    probs  : np.ndarray  shape (N, NUM_CLASSES)  – softmax probabilities
    labels : np.ndarray  shape (N,)              – ground-truth labels
    """
    all_probs  = []
    all_labels = []

    for images, labels in tqdm(loader, desc="  Inference", leave=False):
        images = images.to(device)
        logits = model(images)
        probs  = F.softmax(logits, dim=1)
        all_probs.append(probs.cpu().numpy())
        all_labels.append(labels.numpy())

    return np.vstack(all_probs), np.concatenate(all_labels)


# ─── Visualization ────────────────────────────────────────────────────────────

def plot_confusion_matrix(y_true: np.ndarray,
                          y_pred: np.ndarray,
                          save_path: str | None = None) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ax=ax,
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label",      fontsize=12)
    ax.set_title("Ensemble Confusion Matrix",  fontsize=14)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Ensemble] Confusion matrix saved → {save_path}")
    plt.show()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Ensemble] Device: {device}")

    # ── Test Dataset & Loader ──────────────────────────────────────────────────
    test_ds = HAM10000Dataset(args.test_csv, transform=get_transforms("test"))
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    print(f"[Ensemble] Test samples: {len(test_ds)}")

    # ── Load models & run inference ───────────────────────────────────────────
    model_configs = [
        ("EfficientNet-B0", args.efficientnet_ckpt, _load_efficientnet),
        ("ResNet34",        args.resnet_ckpt,        _load_resnet),
        ("MobileNetV3",     args.mobilenet_ckpt,     _load_mobilenet),
    ]

    all_probs  = []
    true_labels = None

    for model_name, ckpt_path, loader_fn in model_configs:
        ckpt = Path(ckpt_path)
        if not ckpt.exists():
            print(f"[Ensemble] WARNING: '{ckpt_path}' not found – skipping {model_name}")
            continue

        print(f"\n[Ensemble] Running inference with {model_name} …")
        model = loader_fn(ckpt_path, device)
        probs, labels = get_probabilities(model, test_loader, device)

        if true_labels is None:
            true_labels = labels

        # Per-model accuracy
        per_model_preds = probs.argmax(axis=1)
        acc = (per_model_preds == labels).mean()
        print(f"  {model_name} Test Accuracy: {acc:.4f}")

        all_probs.append(probs)
        del model  # free GPU memory

    if not all_probs:
        print("[Ensemble] No model checkpoints found. Train the models first.")
        return

    # ── Soft-voting: average probabilities ────────────────────────────────────
    ensemble_probs = np.mean(np.stack(all_probs, axis=0), axis=0)  # (N, 7)
    ensemble_preds = ensemble_probs.argmax(axis=1)

    ensemble_acc = (ensemble_preds == true_labels).mean()
    print(f"\n[Ensemble] ─── Soft-Voting Ensemble Accuracy: {ensemble_acc:.4f} ───")

    # ── Classification Report ─────────────────────────────────────────────────
    print("\n[Ensemble] Classification Report (Ensemble):")
    print(
        classification_report(
            true_labels,
            ensemble_preds,
            target_names=CLASS_NAMES,
            zero_division=0,
        )
    )

    # ── Confusion Matrix ──────────────────────────────────────────────────────
    plot_confusion_matrix(true_labels, ensemble_preds, save_path=args.save_cm)


if __name__ == "__main__":
    main()
