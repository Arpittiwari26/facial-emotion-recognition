"""
extract_real_fer2013.py

Extracts authentic FER-2013 dataset images from the local Arrow cache into .npy files.
Splits:
  Train: 28,709 samples
  Val:   3,589 samples
  Test:  3,589 samples
"""

import io
import os
import numpy as np
import pyarrow.ipc as ipc
from PIL import Image
from pathlib import Path

CACHE_DIR = Path(r"C:\Users\Arpit\.cache\huggingface\datasets\AutumnQiu___fer2013\default\0.0.0\9c0267afc93fa4d4bfc05d7ff1108a04f72e8392")
SAVE_DIR = Path("dataset")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

def load_arrow_split(filename: str):
    filepath = CACHE_DIR / filename
    print(f"[+] Reading Arrow dataset file: {filepath.name}...")
    reader = ipc.RecordBatchStreamReader(str(filepath))
    table = reader.read_all()
    
    labels_py = table['label'].to_pylist()
    images_col = table['image']
    
    n_samples = len(labels_py)
    print(f"    Extracting {n_samples:,} images...")
    
    X = np.zeros((n_samples, 48, 48, 1), dtype=np.float32)
    y = np.array(labels_py, dtype=np.int64)
    
    for i in range(n_samples):
        img_bytes = images_col[i]['bytes'].as_py()
        img = Image.open(io.BytesIO(img_bytes)).convert('L') # Grayscale
        img_arr = np.array(img, dtype=np.float32) / 255.0
        X[i, :, :, 0] = img_arr
        
    return X, y

def main():
    print("=" * 65)
    print("   EXTRACTING AUTHENTIC FER-2013 DATASET FROM LOCAL CACHE")
    print("=" * 65)
    
    X_train, y_train = load_arrow_split("fer2013-train.arrow")
    X_val, y_val = load_arrow_split("fer2013-valid.arrow")
    X_test, y_test = load_arrow_split("fer2013-test.arrow")
    
    print(f"\n[+] Dataset shapes:")
    print(f"    X_train: {X_train.shape}, dtype: {X_train.dtype}, min: {X_train.min():.4f}, max: {X_train.max():.4f}")
    print(f"    y_train: {y_train.shape}, dtype: {y_train.dtype}")
    print(f"    X_val:   {X_val.shape}")
    print(f"    y_val:   {y_val.shape}")
    print(f"    X_test:  {X_test.shape}")
    print(f"    y_test:  {y_test.shape}")
    
    np.save(SAVE_DIR / "X_train.npy", X_train)
    np.save(SAVE_DIR / "y_train.npy", y_train)
    np.save(SAVE_DIR / "X_val.npy", X_val)
    np.save(SAVE_DIR / "y_val.npy", y_val)
    np.save(SAVE_DIR / "X_test.npy", X_test)
    np.save(SAVE_DIR / "y_test.npy", y_test)
    
    print(f"\n[+] Successfully saved real FER-2013 .npy files to '{SAVE_DIR.resolve()}'")
    print("=" * 65)

if __name__ == "__main__":
    main()
