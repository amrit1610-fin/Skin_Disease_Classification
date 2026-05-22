"""
src/dataset.py
==============
Centralized PyTorch Dataset for HAM10000
-----------------------------------------
Reads from the CSV index files produced by prepare_data.py,
loads images with OpenCV (BGR → RGB), and applies Albumentations
augmentation pipelines:

  Training   : Resize → HorizontalFlip → VerticalFlip → Rotate →
                ShiftScaleRotate → RandomBrightnessContrast →
                HueSaturationValue → Normalize → ToTensorV2
  Val / Test : Resize → Normalize → ToTensorV2

Usage:
    from src.dataset import HAM10000Dataset, get_transforms, LABEL_MAP, CLASS_NAMES

    train_ds = HAM10000Dataset("data_splits/train_split.csv", transform=get_transforms("train"))
    val_ds   = HAM10000Dataset("data_splits/val_split.csv",   transform=get_transforms("val"))
    test_ds  = HAM10000Dataset("data_splits/test_split.csv",  transform=get_transforms("test"))
"""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

import albumentations as A
from albumentations.pytorch import ToTensorV2

# ─── Constants ────────────────────────────────────────────────────────────────

LABEL_MAP = {
    "nv":    0,
    "mel":   1,
    "bkl":   2,
    "bcc":   3,
    "akiec": 4,
    "vasc":  5,
    "df":    6,
}

CLASS_NAMES = [
    "Melanocytic nevi",          # 0
    "Melanoma",                   # 1
    "Benign keratosis-like",      # 2
    "Basal cell carcinoma",       # 3
    "Actinic keratoses",          # 4
    "Vascular lesions",           # 5
    "Dermatofibroma",             # 6
]

NUM_CLASSES = len(LABEL_MAP)

# ImageNet statistics used for normalisation
_MEAN = (0.485, 0.456, 0.406)
_STD  = (0.229, 0.224, 0.225)

IMAGE_SIZE = 224  # Input resolution for all models


# ─── Augmentation Pipelines ───────────────────────────────────────────────────

def get_transforms(split: str) -> A.Compose:
    """
    Return an Albumentations Compose pipeline for the given split.

    Parameters
    ----------
    split : str – one of "train", "val", "test"
    """
    split = split.lower()
    if split == "train":
        return A.Compose([
            A.Resize(IMAGE_SIZE, IMAGE_SIZE),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.Rotate(limit=30, p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=0.1,
                rotate_limit=15,
                border_mode=cv2.BORDER_CONSTANT,
                p=0.5,
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.4,
            ),
            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=20,
                val_shift_limit=10,
                p=0.3,
            ),
            A.Normalize(mean=_MEAN, std=_STD),
            ToTensorV2(),
        ])
    elif split in ("val", "test"):
        return A.Compose([
            A.Resize(IMAGE_SIZE, IMAGE_SIZE),
            A.Normalize(mean=_MEAN, std=_STD),
            ToTensorV2(),
        ])
    else:
        raise ValueError(f"Unknown split '{split}'. Choose from: train, val, test.")


# ─── Dataset Class ────────────────────────────────────────────────────────────

class HAM10000Dataset(Dataset):
    """
    PyTorch Dataset for the HAM10000 Skin Lesion Classification task.

    Parameters
    ----------
    csv_path  : str | Path  – path to one of the split CSV files
    transform : A.Compose   – Albumentations transform pipeline
    """

    def __init__(self, csv_path: str | Path, transform: A.Compose | None = None):
        self.df        = pd.read_csv(csv_path)
        self.transform = transform

        # Validate required columns
        required = {"image_path", "label"}
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(
                f"CSV '{csv_path}' is missing columns: {missing}. "
                "Run src/prepare_data.py first."
            )

        self.image_paths = self.df["image_path"].tolist()
        self.labels      = self.df["label"].tolist()

    # ── Dunder helpers ────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img_path = self.image_paths[idx]
        label    = int(self.labels[idx])

        # Load with OpenCV and convert BGR → RGB
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Image not found at path: '{img_path}'")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform:
            augmented = self.transform(image=img)
            img = augmented["image"]
        else:
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        return img, label

    # ── Utility ───────────────────────────────────────────────────────────────

    def class_counts(self) -> dict:
        """Return a dict mapping label int → count (useful for class weights)."""
        counts = {}
        for lbl in self.labels:
            counts[lbl] = counts.get(lbl, 0) + 1
        return counts

    def compute_class_weights(self) -> torch.Tensor:
        """
        Compute inverse-frequency class weights for use with
        torch.nn.CrossEntropyLoss(weight=...).
        """
        counts = self.class_counts()
        total  = sum(counts.values())
        weights = torch.zeros(NUM_CLASSES)
        for cls_idx in range(NUM_CLASSES):
            cnt = counts.get(cls_idx, 1)
            weights[cls_idx] = total / (NUM_CLASSES * cnt)
        return weights
