"""
src/train_resnet.py
====================
ResNet34 Training Pipeline for HAM10000
-----------------------------------------
- Loads train / val splits from  data_splits/
- Uses the shared HAM10000Dataset + Albumentations transforms
- Fine-tunes a ResNet34 (ImageNet pre-trained) on 7 classes
- Applies class-balanced CrossEntropyLoss
- Uses AdamW optimizer + CosineAnnealingLR scheduler
- Exposes  get_backbone()  so train_resnet_xgboost.py can reuse the encoder
- Saves best checkpoint to:
    checkpoints/resnet_best.pth

Usage:
    python src/train_resnet.py [--epochs 25] [--batch_size 32] [--lr 1e-4]
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import HAM10000Dataset, get_transforms, NUM_CLASSES, CLASS_NAMES

# ─── Configuration ────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train ResNet34 on HAM10000")
    p.add_argument("--train_csv",    default="data_splits/train_split.csv")
    p.add_argument("--val_csv",      default="data_splits/val_split.csv")
    p.add_argument("--checkpoint",   default="checkpoints/resnet_best.pth")
    p.add_argument("--epochs",       type=int,   default=25)
    p.add_argument("--batch_size",   type=int,   default=32)
    p.add_argument("--lr",           type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--num_workers",  type=int,   default=2)
    p.add_argument("--seed",         type=int,   default=42)
    return p.parse_args()


# ─── Model Factory ────────────────────────────────────────────────────────────

def build_model(num_classes: int) -> nn.Module:
    """Load pre-trained ResNet34 and replace the final FC layer."""
    model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
    in_features   = model.fc.in_features
    model.fc      = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes),
    )
    return model


def get_backbone(checkpoint_path: str, device: torch.device) -> nn.Module:
    """
    Load the saved ResNet34 checkpoint and return the backbone
    (everything except the final FC), useful for feature extraction.

    Parameters
    ----------
    checkpoint_path : str   – path to the saved .pth file
    device          : torch.device

    Returns
    -------
    backbone : nn.Module  – ResNet34 without the final classifier
    """
    model = build_model(NUM_CLASSES)
    ckpt  = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device)
    model.eval()

    # Strip the final FC layer to get a flat 512-dim feature extractor
    backbone = nn.Sequential(*list(model.children())[:-1])  # up to avgpool
    return backbone


# ─── Training / Evaluation Loops ─────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device, epoch, total_epochs):
    model.train()
    running_loss = 0.0
    correct      = 0
    total        = 0

    pbar = tqdm(loader, desc=f"[Train] Epoch {epoch+1}/{total_epochs}", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds         = outputs.argmax(dim=1)
        correct      += (preds == labels).sum().item()
        total        += labels.size(0)

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct      = 0
    total        = 0

    for images, labels in tqdm(loader, desc="[Val]  ", leave=False):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss    = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        preds         = outputs.argmax(dim=1)
        correct      += (preds == labels).sum().item()
        total        += labels.size(0)

    return running_loss / total, correct / total


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[ResNet34] Device: {device}")

    torch.manual_seed(args.seed)

    # ── Datasets & Loaders ────────────────────────────────────────────────────
    train_ds = HAM10000Dataset(args.train_csv, transform=get_transforms("train"))
    val_ds   = HAM10000Dataset(args.val_csv,   transform=get_transforms("val"))

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    print(f"[ResNet34] Train samples : {len(train_ds)}")
    print(f"[ResNet34] Val   samples : {len(val_ds)}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(NUM_CLASSES).to(device)

    # ── Loss – class-balanced ─────────────────────────────────────────────────
    class_weights = train_ds.compute_class_weights().to(device)
    criterion     = nn.CrossEntropyLoss(weight=class_weights)

    # ── Optimiser & Scheduler ─────────────────────────────────────────────────
    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    # ── Checkpoint dir ────────────────────────────────────────────────────────
    ckpt_path = Path(args.checkpoint)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_acc = 0.0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, args.epochs
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch [{epoch+1:02d}/{args.epochs}]  "
            f"Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.4f}  |  "
            f"Val Loss: {val_loss:.4f}  Val Acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "epoch":       epoch + 1,
                    "model_state": model.state_dict(),
                    "val_acc":     val_acc,
                    "val_loss":    val_loss,
                    "args":        vars(args),
                },
                ckpt_path,
            )
            print(f"  ✓ Best checkpoint saved → {ckpt_path}  (val_acc={val_acc:.4f})")

    print(f"\n[ResNet34] Training complete. Best val acc: {best_val_acc:.4f}")
    return history


if __name__ == "__main__":
    main()
