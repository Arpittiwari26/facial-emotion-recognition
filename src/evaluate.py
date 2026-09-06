"""
src/evaluate.py

Model evaluation on the test set.

Reports:
  - Test loss, accuracy
  - Per-class precision, recall, F1-score
  - Confusion matrix (saved to results/confusion_matrix.png)
  - Most-confused emotion pairs
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
)

from preprocess import EMOTION_LABELS, load_preprocessed
from tensorflow.keras.models import load_model

RESULTS_DIR = Path("results")


def evaluate(
    model: tf.keras.Model,
    data: dict,
    save_cm: Path = RESULTS_DIR / "confusion_matrix.png",
) -> dict:
    """
    Evaluate the model on the test set.

    Returns dict with metrics for inspection.
    """
    X_test = data["X_test"]
    y_test = data["y_test"]
    label_names = data["label_names"]

    print("\n[evaluate] Evaluating on test set...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
    print(f"\n[evaluate] Test Loss:     {test_loss:.4f}")
    print(f"[evaluate] Test Accuracy: {test_accuracy:.4f}  ({test_accuracy*100:.2f}%)")

    # Get predictions
    print("[evaluate] Generating predictions for confusion matrix...")
    y_pred_probs = model.predict(X_test, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Classification report
    print("\n[evaluate] Classification Report:")
    print("-" * 65)
    report = classification_report(
        y_test, y_pred, target_names=label_names, digits=4
    )
    print(report)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\n[evaluate] Confusion Matrix (raw counts):")
    print(cm)

    # Plot confusion matrix (normalized rows)
    cm_norm = cm.astype(np.float64)
    row_sums = cm_norm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_norm = cm_norm / row_sums
    cm_norm = np.nan_to_num(cm_norm)

    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=label_names,
        yticklabels=label_names,
        vmin=0,
        vmax=1,
        cbar_kws={"label": "Proportion of true class"},
    )
    plt.title("Confusion Matrix (Normalized by True Class)", fontsize=14)
    plt.xlabel("Predicted Emotion", fontsize=12)
    plt.ylabel("True Emotion", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_cm, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n[evaluate] Confusion matrix saved to: {save_cm}")

    # Identify most confused pairs (off-diagonal)
    np.fill_diagonal(cm, 0)  # ignore correct predictions
    flat_idx = np.argmax(cm)
    max_row, max_col = np.unravel_index(flat_idx, cm.shape)
    most_from = label_names[max_row]
    most_to = label_names[max_col]
    most_count = cm[max_row, max_col]
    print(f"\n[evaluate] Most confused pair: {most_from} → {most_to} "
          f"({most_count} misclassifications)")

    # Top-5 confused pairs
    print("[evaluate] Top-5 most confused pairs:")
    flat_indices = np.argsort(cm.flatten())[::-1][:10]
    for idx in flat_indices:
        r, c = np.unravel_index(idx, cm.shape)
        if cm[r, c] > 0:
            print(f"    {label_names[r]:<10} → {label_names[c]:<10}: "
                  f"{cm[r, c]:4d} times")

    return {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "y_true": y_test,
        "y_pred": y_pred,
        "y_pred_probs": y_pred_probs,
        "confusion_matrix": cm,
    }


def cli():
    parser = argparse.ArgumentParser(
        description="Evaluate trained FER model on test set"
    )
    parser.add_argument("--model", type=str,
                        default="models/emotion_model.keras",
                        help="Path to trained .keras model")
    parser.add_argument("--results-dir", type=str, default="results")
    args = parser.parse_args()

    print("=" * 60)
    print("  Facial Emotion Recognition — Evaluation")
    print("=" * 60)

    model = load_model(args.model)
    data = load_preprocessed()
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    results = evaluate(
        model, data,
        save_cm=results_dir / "confusion_matrix.png",
    )

    print(f"\n[evaluate] Summary:")
    print(f"  Test Accuracy : {results['test_accuracy']:.4f}  "
          f"({results['test_accuracy']*100:.2f}%)")
    print(f"  Test Loss     : {results['test_loss']:.4f}")

    # Check training plots
    hist_png = results_dir / "training_history.png"
    if hist_png.exists():
        print(f"\n[evaluate] Training plots: {hist_png}")
    else:
        print(f"\n[evaluate] Note: training plots not found at {hist_png}")
        print("           Run src/train.py first to generate them.")


if __name__ == "__main__":
    cli()
