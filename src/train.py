"""
src/train.py

Training script for the FER-2013 emotion CNN.

Steps:
  1. Load preprocessed .npy data
  2. Build the CNN model
  3. Compile with Adam + sparse categorical crossentropy
  4. Set up callbacks: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
  5. Train
  6. Save plots to results/
  7. Save best model as models/emotion_model.keras
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving plots
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from preprocess import load_preprocessed
from model import build_emotion_model, compile_model

# ── Configuration ───────────────────────────────────────────────────
BATCH_SIZE = 64
EPOCHS = 80
LEARNING_RATE = 1e-3
SEED = 42

# Dirs
MODELS_DIR = Path("models")
RESULTS_DIR = Path("results")
MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def check_gpu():
    """Print GPU availability status."""
    gpus = tf.config.list_physical_devices("GPU")
    print(f"\n[train] TensorFlow GPU available: {len(gpus) > 0}")
    if gpus:
        for gpu in gpus:
            print(f"[train]   GPU device: {gpu.name}")
        # Allow memory growth so TF doesn't grab all VRAM at once
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        print("[train]   No GPU detected. Training on CPU.")
        print("[train]   (Training 48x48 grayscale on CPU is feasible.)")
    return len(gpus) > 0


def build_callbacks(model_path: Path):
    """
    Create training callbacks.

    EarlyStopping:
      - Stops training when val_loss stops improving.
      - patience=12: wait 12 epochs without improvement before stopping.
      - restore_best_weights=True: roll back to the best epoch's weights.
      - Why: prevents overfitting. The model keeps training on training data
        but validation loss rises → the model is memorizing, not learning.

    ReduceLROnPlateau:
      - Halves the learning rate when val_loss plateaus.
      - patience=6: wait 6 epochs before reducing LR.
      - factor=0.5: multiply LR by 0.5.
      - Why: A high learning rate helps explore the loss landscape early.
        When progress stalls, lowering LR lets the optimizer take smaller,
        more precise steps toward a better minimum.

    ModelCheckpoint:
      - Saves the model whenever val_loss improves.
      - save_best_only=True: never overwrites with a worse model.
      - Why: training is noisy. The best model may be at epoch 30 even if
        training runs to epoch 60. We want to keep the best one.
    """
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=12,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=6,
            min_lr=1e-6,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(model_path),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
    ]


def train(
    data: dict,
    model_path: str | Path = "models/emotion_model.keras",
    batch_size: int = BATCH_SIZE,
    epochs: int = EPOCHS,
    learning_rate: float = LEARNING_RATE,
    seed: int = SEED,
) -> tuple[tf.keras.Model, dict]:
    """
    Train the emotion CNN.

    Returns (model, history_dict).
    """
    # Reproducibility
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # Check GPU
    check_gpu()

    # Build & compile model
    print(f"\n[train] Building model...")
    print(f"[train] Input shape: {data['X_train'].shape[1:]}")
    print(f"[train] Training samples: {data['X_train'].shape[0]}")
    print(f"[train] Validation samples: {data['X_val'].shape[0]}")
    print(f"[train] Batch size: {batch_size}")
    print(f"[train] Max epochs: {epochs}")
    print(f"[train] Learning rate: {learning_rate}\n")

    model = build_emotion_model(input_shape=data["X_train"].shape[1:])
    model = compile_model(model, learning_rate=learning_rate)

    # Print a compact model summary
    model.summary()

    # Count parameters
    total_params = model.count_params()
    print(f"\n[train] Total trainable parameters: {total_params:,}")

    # Data augmentation (applied only to training data)
    # These layers operate on GPU and are inactive during inference.
    data_augmentation = tf.keras.Sequential(
        name="data_augmentation",
        layers=[
            tf.keras.layers.RandomFlip("horizontal", seed=seed),
            tf.keras.layers.RandomRotation(factor=0.1, seed=seed),  # ±18°
            tf.keras.layers.RandomZoom(
                height_factor=(-0.1, 0.1),
                width_factor=(-0.1, 0.1),
                seed=seed,
            ),
            tf.keras.layers.RandomTranslation(
                height_factor=(-0.05, 0.05),
                width_factor=(-0.05, 0.05),
                seed=seed,
            ),
        ],
    )

    callbacks = build_callbacks(Path(model_path))

    print("[train] Starting training...")
    print("-" * 60)

    start_time = time.time()

    # Train with data augmentation applied to training data only
    # We use a tf.data pipeline for efficient augmentation
    train_dataset = tf.data.Dataset.from_tensor_slices(
        (data["X_train"], data["y_train"])
    )
    train_dataset = train_dataset.shuffle(buffer_size=len(data["X_train"]),
                                          seed=seed)
    train_dataset = train_dataset.batch(batch_size)
    train_dataset = train_dataset.map(
        lambda x, y: (data_augmentation(x, training=True), y),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

    val_dataset = tf.data.Dataset.from_tensor_slices(
        (data["X_val"], data["y_val"])
    ).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # Compute smoothed class weights to handle FER-2013 class imbalance smoothly
    from sklearn.utils.class_weight import compute_class_weight
    classes = np.unique(data["y_train"])
    cw_arr = compute_class_weight(class_weight="balanced", classes=classes, y=data["y_train"])
    cw_smoothed = np.power(cw_arr, 0.5)
    class_weight_dict = dict(zip(classes, cw_smoothed))

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=callbacks,
        class_weight=class_weight_dict,
        verbose=1,
    )

    elapsed = time.time() - start_time
    print("-" * 60)
    print(f"\n[train] Training completed in {elapsed:.1f}s ({elapsed/60:.1f} min)")

    # Load the best checkpoint
    best_model_path = Path(model_path)
    if best_model_path.exists():
        print(f"[train] Loading best model from {best_model_path}")
        model = tf.keras.models.load_model(best_model_path)

    # Plot training history
    history_dict = history.history if hasattr(history, "history") else history
    plot_training_history(history_dict, RESULTS_DIR / "training_history.png")

    # Save history as JSON for later use (e.g., in Gradio app)
    import json
    with open(RESULTS_DIR / "training_history.json", "w") as f:
        json.dump(history_dict, f, indent=2)
    print(f"[train] History saved to {RESULTS_DIR / 'training_history.json'}")

    return model, history_dict


def plot_training_history(history: dict, save_path: Path):
    """Plot training vs validation accuracy and loss, save to file."""
    acc = history["accuracy"]
    val_acc = history["val_accuracy"]
    loss = history["loss"]
    val_loss = history["val_loss"]
    epochs_range = range(1, len(acc) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # ── Accuracy plot ──
    ax1.plot(epochs_range, acc, "b-", label="Training Accuracy", linewidth=2)
    ax1.plot(epochs_range, val_acc, "r-", label="Validation Accuracy", linewidth=2)
    ax1.set_title("Training vs Validation Accuracy", fontsize=14)
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Mark best validation accuracy epoch
    best_epoch = int(np.argmax(val_acc)) + 1
    best_val_acc = max(val_acc)
    ax1.axvline(x=best_epoch, color="g", linestyle="--", alpha=0.5,
                label=f"Best val acc: epoch {best_epoch} ({best_val_acc:.4f})")
    ax1.legend()

    # ── Loss plot ──
    ax2.plot(epochs_range, loss, "b-", label="Training Loss", linewidth=2)
    ax2.plot(epochs_range, val_loss, "r-", label="Validation Loss", linewidth=2)
    ax2.set_title("Training vs Validation Loss", fontsize=14)
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Mark best validation loss epoch
    best_loss_epoch = int(np.argmin(val_loss)) + 1
    best_val_loss = min(val_loss)
    ax2.axvline(x=best_loss_epoch, color="g", linestyle="--", alpha=0.5,
                label=f"Best val loss: epoch {best_loss_epoch} ({best_val_loss:.4f})")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[train] Training plots saved to: {save_path}")

    # Also print interpretation guide
    print("\n[train] How to read these plots:")
    print("  - Good convergence:")
    print("      Both training and validation accuracy rise together and plateau.")
    print("      Both losses decrease together and plateau. Gap between them is small.")
    print("  - Overfitting:")
    print("      Training accuracy keeps rising but validation accuracy plateaus or drops.")
    print("      Training loss keeps falling but validation loss rises after some epoch.")
    print("      A visible gap opens between the two curves.")
    print("  - Underfitting:")
    print("      Both training and validation accuracy stay low.")
    print("      Both losses stay high. Adding more layers/epochs may help.")
    print(f"  - Best epoch (by val_loss): {int(np.argmin(val_loss)) + 1}")
    print(f"  - Best val accuracy: {max(val_acc):.4f} at epoch {int(np.argmax(val_acc)) + 1}")


def cli():
    parser = argparse.ArgumentParser(description="Train FER emotion CNN")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--model-path", type=str,
                        default="models/emotion_model.keras")
    args = parser.parse_args()

    print("=" * 60)
    print("  Facial Emotion Recognition — Training")
    print("=" * 60)

    data = load_preprocessed()
    model, history = train(
        data,
        model_path=args.model_path,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        seed=args.seed,
    )
    print("\n[train] Done. Model saved to", args.model_path)


if __name__ == "__main__":
    cli()
