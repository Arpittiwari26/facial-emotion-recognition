"""
src/preprocess.py

Data preprocessing for FER-2013 dataset.

FER-2013 is provided as a CSV file (usually from Kaggle).
Each row: emotion_label, pixels (space-separated 2304 grayscale values), Usage

This script:
1. Loads the CSV
2. Reshapes pixel strings into 48x48 images
3. Splits into train/val/test by Usage column
4. Normalizes pixel values to [0, 1]
5. Saves as .npy files for fast loading during training
"""

from __future__ import annotations

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


EMOTION_LABELS = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Sad",
    "Surprise",
    "Neutral",
]

NUM_CLASSES = len(EMOTION_LABELS)
IMG_SIZE = 48


def load_fer_csv(csv_path: str | Path) -> pd.DataFrame:
    """Load FER-2013 CSV. Raises FileNotFoundError if missing."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"FER-2013 CSV not found at {csv_path}\n"
            "Download fer2013.csv from Kaggle and place it here."
        )
    df = pd.read_csv(csv_path)
    expected_cols = {"emotion", "pixels", "Usage"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")
    print(f"[preprocess] CSV loaded: {len(df)} rows")
    print(f"[preprocess] Usage distribution:\n{df['Usage'].value_counts()}")
    return df


def parse_pixels(pixels_str: str) -> np.ndarray:
    """Convert space-separated pixel string into a 48x48 numpy array."""
    pixel_values = np.array([int(p) for p in pixels_str.split()], dtype=np.float64)
    if len(pixel_values) != IMG_SIZE * IMG_SIZE:
        raise ValueError(f"Expected {IMG_SIZE*IMG_SIZE} pixels, got {len(pixel_values)}")
    return pixel_values.reshape(IMG_SIZE, IMG_SIZE)


def preprocess(
    csv_path: str | Path = "dataset/fer2013.csv",
    save_dir: str | Path = "dataset",
    test_size: float = 0.15,
    random_state: int = 42,
) -> dict:
    """
    Full preprocessing pipeline.

    Steps:
    1. Load CSV
    2. Parse pixels into image arrays
    3. Normalize: pixel / 255.0 → values in [0, 1]
    4. Split:
       - 'Training' rows → train (85%) + val (15%)
       - 'PublicTest' rows → test set
    5. Add channel dimension: (N, 48, 48) → (N, 48, 48, 1)
    6. Save as .npy files
    """
    df = load_fer_csv(csv_path)

    print("[preprocess] Parsing pixel strings into images...")
    images = np.array([parse_pixels(p) for p in df["pixels"].values])
    images = images / 255.0  # Normalize to [0, 1]
    labels = df["emotion"].values.astype(np.int64)

    train_mask = df["Usage"].str.strip() == "Training"
    test_mask = df["Usage"].str.strip() == "PublicTest"

    X_train_full = images[train_mask]
    y_train_full = labels[train_mask]
    X_test = images[test_mask]
    y_test = labels[test_mask]

    # Split training data into train + validation (stratified)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full,
        test_size=test_size, random_state=random_state,
        stratify=y_train_full,
    )

    print(f"[preprocess] Split complete:")
    print(f"  Train: {X_train.shape[0]} | Val: {X_val.shape[0]} | Test: {X_test.shape[0]}")
    print(f"  Image shape: {X_train.shape[1:]} (grayscale)")

    # Add channel axis for Keras Conv2D: (N, H, W) → (N, H, W, 1)
    X_train = np.expand_dims(X_train, axis=-1)
    X_val = np.expand_dims(X_val, axis=-1)
    X_test = np.expand_dims(X_test, axis=-1)
    print(f"  Final shape with channel: {X_train.shape[1:]}")

    # Save .npy files
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    for name, arr in [
        ("X_train", X_train), ("y_train", y_train),
        ("X_val", X_val), ("y_val", y_val),
        ("X_test", X_test), ("y_test", y_test),
    ]:
        np.save(save_dir / f"{name}.npy", arr)

    print(f"[preprocess] Saved .npy files to {save_dir}/")
    return {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
        "label_names": EMOTION_LABELS,
        "num_classes": NUM_CLASSES,
        "img_size": IMG_SIZE,
    }


def load_preprocessed(save_dir: str | Path = "dataset") -> dict:
    """Load preprocessed .npy files from disk."""
    save_dir = Path(save_dir)
    required = ["X_train.npy", "y_train.npy", "X_val.npy", "y_val.npy",
                "X_test.npy", "y_test.npy"]
    for f in required:
        if not (save_dir / f).exists():
            raise FileNotFoundError(
                f"Missing {f} in {save_dir}. Run preprocess() first "
                "(python -m src.preprocess) to generate the dataset."
            )
    X_train = np.load(save_dir / "X_train.npy")
    y_train = np.load(save_dir / "y_train.npy")
    X_val = np.load(save_dir / "X_val.npy")
    y_val = np.load(save_dir / "y_val.npy")
    X_test = np.load(save_dir / "X_test.npy")
    y_test = np.load(save_dir / "y_test.npy")

    print(f"[preprocess] Loaded from disk: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")
    return {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
        "label_names": EMOTION_LABELS,
        "num_classes": NUM_CLASSES,
        "img_size": IMG_SIZE,
    }


def cli():
    parser = argparse.ArgumentParser(description="FER-2013 preprocessing")
    parser.add_argument("--csv", type=str, default="dataset/fer2013.csv")
    parser.add_argument("--save-dir", type=str, default="dataset")
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()
    preprocess(csv_path=args.csv, save_dir=args.save_dir,
               test_size=args.test_size, random_state=args.random_state)


if __name__ == "__main__":
    cli()
