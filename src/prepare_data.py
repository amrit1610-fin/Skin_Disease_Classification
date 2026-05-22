"""
src/prepare_data.py
====================
HAM10000 Data Preprocessor
---------------------------
- Auto-detects the HAM10000 folder under /kaggle/input/datasets/kmader
- Parses HAM10000_metadata.csv and matches each image ID to its file path
  across ham10000_images_part_1 and ham10000_images_part_2
- Performs a stratified 70 / 15 / 15 train / val / test split
- Applies RandomOverSampler **only** on the training fold
- Saves three lean CSV index files into  data_splits/
    data_splits/train_split.csv
    data_splits/val_split.csv
    data_splits/test_split.csv

Usage (standalone):
    python src/prepare_data.py                      # default Kaggle path
    python src/prepare_data.py --root /path/to/data # custom root
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler

# ─── Label Mapping ────────────────────────────────────────────────────────────
LABEL_MAP = {
    "nv":   0,  # Melanocytic nevi
    "mel":  1,  # Melanoma
    "bkl":  2,  # Benign keratosis-like lesions
    "bcc":  3,  # Basal cell carcinoma
    "akiec": 4, # Actinic keratoses / Intraepithelial carcinoma
    "vasc": 5,  # Vascular lesions
    "df":   6,  # Dermatofibroma
}

CLASS_NAMES = [
    "Melanocytic nevi",
    "Melanoma",
    "Benign keratosis-like lesions",
    "Basal cell carcinoma",
    "Actinic keratoses",
    "Vascular lesions",
    "Dermatofibroma",
]

NUM_CLASSES = len(LABEL_MAP)

# ─── Helpers ──────────────────────────────────────────────────────────────────

def _find_kmader_root(base: Path) -> Path:
    """Walk base looking for the kmader dataset folder."""
    if (base / "HAM10000_metadata.csv").exists():
        return base
    for candidate in base.rglob("HAM10000_metadata.csv"):
        return candidate.parent
    raise FileNotFoundError(
        f"Could not find HAM10000_metadata.csv under {base}. "
        "Please pass the correct --root path."
    )


def _build_image_index(kmader_root: Path) -> dict:
    """Return {image_id: absolute_path} scanning part_1 and part_2."""
    image_dirs = [
        kmader_root / "ham10000_images_part_1",
        kmader_root / "ham10000_images_part_2",
    ]
    # Also support flat layout (all images in root)
    image_dirs.append(kmader_root)

    index = {}
    for img_dir in image_dirs:
        if not img_dir.is_dir():
            continue
        for fpath in img_dir.glob("*.jpg"):
            image_id = fpath.stem          # e.g. ISIC_0024306
            index[image_id] = str(fpath)
        for fpath in img_dir.glob("*.png"):
            image_id = fpath.stem
            if image_id not in index:
                index[image_id] = str(fpath)
    return index


# ─── Main Preparation Logic ───────────────────────────────────────────────────

def prepare(root: str, output_dir: str = "data_splits") -> tuple:
    """
    Full preparation pipeline.

    Parameters
    ----------
    root       : str  – path to the kmader dataset root (contains metadata CSV)
    output_dir : str  – where to write the three split CSVs

    Returns
    -------
    train_df, val_df, test_df  (pandas DataFrames)
    """
    base_path   = Path(root)
    kmader_root = _find_kmader_root(base_path)
    print(f"[prepare_data] Dataset root  : {kmader_root}")

    # ── 1. Load metadata ──────────────────────────────────────────────────────
    meta_csv = kmader_root / "HAM10000_metadata.csv"
    df = pd.read_csv(meta_csv)
    print(f"[prepare_data] Metadata rows  : {len(df)}")

    # ── 2. Build image index and match paths ──────────────────────────────────
    img_index = _build_image_index(kmader_root)
    print(f"[prepare_data] Images found   : {len(img_index)}")

    df["image_path"] = df["image_id"].map(img_index)
    missing = df["image_path"].isna().sum()
    if missing > 0:
        print(f"[prepare_data] WARNING: {missing} image IDs had no matching file – dropping them.")
        df = df.dropna(subset=["image_path"])

    # ── 3. Encode labels ──────────────────────────────────────────────────────
    df["label"] = df["dx"].map(LABEL_MAP)
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    # Keep only the columns we need
    df = df[["image_id", "image_path", "dx", "label"]].reset_index(drop=True)

    print("\n[prepare_data] Class distribution (before oversampling):")
    for cls, cnt in df["dx"].value_counts().items():
        print(f"  {cls:8s}: {cnt}")

    # ── 4. Stratified split: train 70 / (val+test) 30 ────────────────────────
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        stratify=df["label"],
        random_state=42,
    )
    # Split the remaining 30% into val 15% / test 15% (each is 50% of temp)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        stratify=temp_df["label"],
        random_state=42,
    )

    print(f"\n[prepare_data] Split sizes (before oversampling):")
    print(f"  Train : {len(train_df)}")
    print(f"  Val   : {len(val_df)}")
    print(f"  Test  : {len(test_df)}")

    # ── 5. Oversample training fold only ─────────────────────────────────────
    ros = RandomOverSampler(random_state=42)
    # ROS needs a 2-D feature array – we use a dummy column
    X_train = train_df[["image_id", "image_path", "dx", "label"]].copy()
    y_train = train_df["label"].values

    X_res, y_res = ros.fit_resample(X_train, y_train)
    train_df = pd.DataFrame(X_res, columns=["image_id", "image_path", "dx", "label"])
    train_df["label"] = train_df["label"].astype(int)
    train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle

    print(f"\n[prepare_data] Train size (after oversampling): {len(train_df)}")
    print("[prepare_data] Class distribution (train, after oversampling):")
    for cls, cnt in train_df["dx"].value_counts().items():
        print(f"  {cls:8s}: {cnt}")

    # ── 6. Save CSVs ──────────────────────────────────────────────────────────
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(out_dir / "train_split.csv", index=False)
    val_df.to_csv(out_dir   / "val_split.csv",   index=False)
    test_df.to_csv(out_dir  / "test_split.csv",  index=False)

    print(f"\n[prepare_data] Saved splits to '{out_dir}/'")
    print(f"  train_split.csv : {len(train_df)} rows")
    print(f"  val_split.csv   : {len(val_df)} rows")
    print(f"  test_split.csv  : {len(test_df)} rows")

    return train_df, val_df, test_df


# ─── CLI Entry Point ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare HAM10000 data splits.")
    parser.add_argument(
        "--root",
        type=str,
        default="/kaggle/input/datasets/kmader",
        help="Path to the kmader dataset root directory.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data_splits",
        help="Directory to save the split CSV files.",
    )
    args = parser.parse_args()
    prepare(root=args.root, output_dir=args.output_dir)
