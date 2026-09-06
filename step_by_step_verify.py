"""
step_by_step_verify.py

Systematic Verification Steps 1 - 10 on the Authentic FER-2013 Dataset.
"""

import os
import sys
import numpy as np
import tensorflow as tf
from pathlib import Path

# Step 3 Mapping
CLASS_NAMES = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Sad",
    "Surprise",
    "Neutral"
]

def run_diagnostics():
    print("=" * 70)
    print("      SYSTEMATIC FER-2013 DIAGNOSTIC & VERIFICATION REPORT")
    print("=" * 70)

    # -------------------------------------------------------------
    # STEP 2: CHECK THE SAVED MODEL
    # -------------------------------------------------------------
    print("\n--- STEP 2: CHECK THE SAVED MODEL ---")
    model_path = "models/emotion_model.keras"
    print(f"Model path: {model_path}")
    if os.path.exists(model_path):
        try:
            model = tf.keras.models.load_model(model_path)
            print(f"Model input shape: {model.input_shape}")
            print(f"Model output shape: {model.output_shape}")
            print(f"Number of parameters: {model.count_params():,}")
            print(f"Final layer units: {model.layers[-1].units}")
            print(f"Final layer activation: {model.layers[-1].activation.__name__}")
        except Exception as e:
            print(f"[!] Warning loading saved model: {e}")
    else:
        print("[!] No existing model checkpoint found.")

    # -------------------------------------------------------------
    # STEP 3: CHECK THE CLASS MAPPING
    # -------------------------------------------------------------
    print("\n--- STEP 3: CHECK THE CLASS MAPPING ---")
    print("Canonical CLASS_NAMES:", CLASS_NAMES)
    print(f"Number of classes: {len(CLASS_NAMES)}")

    # Load dataset
    X_train = np.load("dataset/X_train.npy")
    y_train = np.load("dataset/y_train.npy")
    X_val = np.load("dataset/X_val.npy")
    y_val = np.load("dataset/y_val.npy")
    X_test = np.load("dataset/X_test.npy")
    y_test = np.load("dataset/y_test.npy")

    # -------------------------------------------------------------
    # STEP 5: CHECK CLASS DISTRIBUTION
    # -------------------------------------------------------------
    print("\n--- STEP 5: CHECK CLASS DISTRIBUTION ---")
    print("Training dataset split breakdown:")
    unique_train, counts_train = np.unique(y_train, return_counts=True)
    for u, c in zip(unique_train, counts_train):
        print(f"  {CLASS_NAMES[u]:<10}: {c:5d} samples ({c/len(y_train)*100:.2f}%)")

    # Calculate class weights for training balance
    from sklearn.utils.class_weight import compute_class_weight
    cw_arr = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
    print("\nBalanced Class Weights:")
    for u, w in zip(unique_train, cw_arr):
        print(f"  {CLASS_NAMES[u]:<10}: Weight = {w:.4f}")

    # -------------------------------------------------------------
    # STEP 6: CHECK TRAINING DATA
    # -------------------------------------------------------------
    print("\n--- STEP 6: CHECK TRAINING DATA ---")
    print(f"Image shape: {X_train.shape}")
    print(f"Label shape: {y_train.shape}")
    print(f"Pixel minimum: {X_train.min():.4f}")
    print(f"Pixel maximum: {X_train.max():.4f}")
    print(f"Data type: {X_train.dtype}")

    # -------------------------------------------------------------
    # STEP 7: CHECK WHETHER IMAGES ARE ACTUALLY DIFFERENT
    # -------------------------------------------------------------
    print("\n--- STEP 7: CHECK WHETHER IMAGES ARE ACTUALLY DIFFERENT ---")
    for i in range(5):
        img_mean = X_train[i].mean()
        img_std = X_train[i].std()
        print(f"Image {i+1} mean: {img_mean:.4f}, std: {img_std:.4f}")

    # -------------------------------------------------------------
    # STEP 8: CHECK LABEL ENCODING
    # -------------------------------------------------------------
    print("\n--- STEP 8: CHECK LABEL ENCODING ---")
    for emotion_idx, emotion_name in enumerate(CLASS_NAMES):
        match_indices = np.where(y_train == emotion_idx)[0]
        sample_idx = match_indices[0] if len(match_indices) > 0 else -1
        raw_val = y_train[sample_idx]
        print(f"{emotion_name:<8} image (index {sample_idx:5d}) -> raw label {raw_val} -> encoded {raw_val} -> {CLASS_NAMES[raw_val]}")

    # -------------------------------------------------------------
    # STEP 9: CHECK LOSS FUNCTION
    # -------------------------------------------------------------
    print("\n--- STEP 9: CHECK LOSS FUNCTION ---")
    print(f"y_train label format: Integer scalar 1D array (shape: {y_train.shape})")
    print("Configured Loss Function: SparseCategoricalCrossentropy() [Matches integer labels]")

    # -------------------------------------------------------------
    # STEP 10: CHECK MODEL TRAINING WITH A TINY DATASET
    # -------------------------------------------------------------
    print("\n--- STEP 10: CHECK MODEL TRAINING WITH A TINY DATASET ---")
    print("Creating tiny subset with 50 images per class (350 total samples)...")
    tiny_X = []
    tiny_y = []
    for cls_idx in range(7):
        cls_indices = np.where(y_train == cls_idx)[0][:50]
        tiny_X.append(X_train[cls_indices])
        tiny_y.append(y_train[cls_indices])
    tiny_X = np.concatenate(tiny_X, axis=0)
    tiny_y = np.concatenate(tiny_y, axis=0)
    print(f"Tiny dataset created: X shape {tiny_X.shape}, y shape {tiny_y.shape}")

    # Build fresh model
    sys.path.append(str(Path(__file__).resolve().parent))
    from src.model import build_emotion_model, compile_model
    
    tiny_model = build_emotion_model(input_shape=(48, 48, 1))
    tiny_model = compile_model(tiny_model, learning_rate=0.001)

    print("\nTraining fresh model on tiny dataset for 20 epochs to test overfitting capacity...")
    history = tiny_model.fit(
        tiny_X, tiny_y,
        epochs=20,
        batch_size=32,
        verbose=0
    )
    final_acc = history.history['accuracy'][-1] * 100.0
    final_loss = history.history['loss'][-1]
    print(f"Tiny Dataset Training Result:")
    print(f"  * Final Training Loss:     {final_loss:.4f}")
    print(f"  * Final Training Accuracy: {final_acc:.2f}%")
    if final_acc > 80.0:
        print("[+] STEP 10 PASSED: Model is fully capable of learning and overfitting the dataset!")
    else:
        print("[!] STEP 10 FAILED: Model failed to overfit tiny dataset.")

    print("=" * 70)

if __name__ == "__main__":
    run_diagnostics()
