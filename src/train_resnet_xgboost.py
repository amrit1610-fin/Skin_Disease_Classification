"""
src/train_resnet_xgboost.py
============================
ResNet34 + XGBoost Hybrid Pipeline for HAM10000
-------------------------------------------------
Pipeline:
  1. Load the frozen, pre-trained ResNet34 backbone from checkpoints/resnet_best.pth
  2. Pass train and validation images through the backbone to extract flat 512-dim
     feature embeddings
  3. Train an XGBClassifier on those extracted features
  4. Evaluate on the validation set and print a classification report
  5. Save the trained XGBoost model to checkpoints/resnet_xgboost.json

Usage:
    python src/train_resnet_xgboost.py [--train_csv ...] [--val_csv ...]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import classification_report
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import HAM10000Dataset, get_transforms, NUM_CLASSES, CLASS_NAMES
from src.train_resnet import get_backbone

# ─── Configuration ────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="ResNet34 + XGBoost Hybrid on HAM10000")
    p.add_argument("--train_csv",      default="data_splits/train_split.csv")
    p.add_argument("--val_csv",        default="data_splits/val_split.csv")
    p.add_argument("--resnet_ckpt",    default="checkpoints/resnet_best.pth")
    p.add_argument("--xgb_output",     default="checkpoints/resnet_xgboost.json")
    p.add_argument("--batch_size",     type=int, default=64)
    p.add_argument("--num_workers",    type=int, default=2)
    p.add_argument("--n_estimators",   type=int, default=500)
    p.add_argument("--max_depth",      type=int, default=6)
    p.add_argument("--learning_rate",  type=float, default=0.1)
    p.add_argument("--seed",           type=int, default=42)
    return p.parse_args()


# ─── Feature Extraction ───────────────────────────────────────────────────────

@torch.no_grad()
def extract_features(backbone: torch.nn.Module,
                     loader: DataLoader,
                     device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """
    Run all batches through the frozen backbone and return:
      features : np.ndarray  shape (N, 512)
      labels   : np.ndarray  shape (N,)
    """
    all_feats  = []
    all_labels = []

    backbone.eval()
    for images, labels in tqdm(loader, desc="Extracting features", leave=False):
        images = images.to(device)
        # backbone output: (B, 512, 1, 1) – flatten to (B, 512)
        feats  = backbone(images).squeeze(-1).squeeze(-1)
        all_feats.append(feats.cpu().numpy())
        all_labels.append(labels.numpy())

    return np.vstack(all_feats), np.concatenate(all_labels)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[ResNet+XGB] Device: {device}")

    # ── 1. Load frozen ResNet34 backbone ──────────────────────────────────────
    resnet_ckpt = Path(args.resnet_ckpt)
    if not resnet_ckpt.exists():
        raise FileNotFoundError(
            f"ResNet checkpoint not found at '{resnet_ckpt}'. "
            "Run  python src/train_resnet.py  first."
        )
    backbone = get_backbone(str(resnet_ckpt), device)
    print(f"[ResNet+XGB] Backbone loaded from: {resnet_ckpt}")

    # ── 2. Datasets & Loaders – no augmentation for feature extraction ─────
    train_ds = HAM10000Dataset(args.train_csv, transform=get_transforms("val"))
    val_ds   = HAM10000Dataset(args.val_csv,   transform=get_transforms("val"))

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    print(f"[ResNet+XGB] Train samples : {len(train_ds)}")
    print(f"[ResNet+XGB] Val   samples : {len(val_ds)}")

    # ── 3. Extract embeddings ─────────────────────────────────────────────────
    print("\n[ResNet+XGB] Extracting training features …")
    X_train, y_train = extract_features(backbone, train_loader, device)

    print("[ResNet+XGB] Extracting validation features …")
    X_val,   y_val   = extract_features(backbone, val_loader,   device)

    print(f"[ResNet+XGB] Feature shape – Train: {X_train.shape}, Val: {X_val.shape}")

    # ── 4. Train XGBClassifier ────────────────────────────────────────────────
    print("\n[ResNet+XGB] Training XGBClassifier …")
    xgb = XGBClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        objective="multi:softmax",
        num_class=NUM_CLASSES,
        use_label_encoder=False,
        eval_metric="mlogloss",
        tree_method="gpu_hist" if torch.cuda.is_available() else "hist",
        random_state=args.seed,
        n_jobs=-1,
    )
    xgb.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=50,
    )

    # ── 5. Evaluate ───────────────────────────────────────────────────────────
    y_pred = xgb.predict(X_val)
    print("\n[ResNet+XGB] Validation Classification Report:")
    print(
        classification_report(
            y_val, y_pred,
            target_names=CLASS_NAMES,
            zero_division=0,
        )
    )

    # ── 6. Save XGBoost model ─────────────────────────────────────────────────
    out_path = Path(args.xgb_output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    xgb.save_model(str(out_path))
    print(f"[ResNet+XGB] XGBoost model saved → {out_path}")


if __name__ == "__main__":
    main()
