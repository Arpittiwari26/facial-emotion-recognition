"""
src/gradcam.py

Grad-CAM (Gradient-weighted Class Activation Mapping) for the emotion CNN.

Grad-CAM highlights the regions of the input face image that most influenced
the model's prediction for a given emotion class.

How it works:
  1. Forward pass: get the final conv layer's feature maps and the predictions.
  2. Backward pass: compute gradients of the target class score w.r.t. the
     feature maps.
  3. Weight each feature map by the global average of its gradient.
  4. Compute a weighted sum of feature maps → coarse heatmap.
  5. ReLU the heatmap (we only care about positive contributions).
  6. Resize heatmap to input size and overlay on the original image.

Caveat:
  Grad-CAM shows where the model LOOKS, not what biologically causes the
  emotion. A heatmap on the mouth for "Happy" means the model used the
  mouth region for its decision — it does NOT prove the mouth causes happiness.

Reference:
  Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks
  via Gradient-based Localization", ICCV 2017.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf


def find_conv_layer(model: tf.keras.Model) -> tf.keras.layers.Layer | None:
    """Find the last Conv2D layer in the model (for Grad-CAM)."""
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer
    return None


def compute_gradcam(
    model: tf.keras.Model,
    face_input: np.ndarray,
    target_class: int | None = None,
) -> tuple[np.ndarray, str]:
    """
    Compute Grad-CAM heatmap for a preprocessed face input.

    Args:
      model: trained Keras model.
      face_input: preprocessed face, shape (1, 48, 48, 1), values in [0,1].
      target_class: emotion index to explain. If None, uses predicted class.

    Returns:
      heatmap: 2D numpy array (48, 48), values in [0, 1].
      explained_class: emotion label string.
    """
    conv_layer = find_conv_layer(model)
    if conv_layer is None:
        raise ValueError("No Conv2D layer found in model. Grad-CAM requires a conv layer.")

    # Build a model that outputs both the conv feature maps and the predictions
    grad_model = tf.keras.Model(
        inputs=model.input,
        outputs=[conv_layer.output, model.output],
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(face_input, training=False)
        if target_class is None:
            target_class = int(np.argmax(predictions[0]))
        class_score = predictions[:, target_class]

    # Gradients of the class score w.r.t. conv feature maps
    grads = tape.gradient(class_score, conv_outputs)

    # Global average pooling of gradients → weight per feature map
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weighted combination of feature maps
    conv_outputs = conv_outputs[0]  # remove batch dim → (H, W, C)
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)

    # ReLU: keep only positive contributions
    heatmap = tf.nn.relu(heatmap)

    # Normalize to [0, 1]
    max_val = tf.reduce_max(heatmap)
    if max_val > 0:
        heatmap = heatmap / max_val
    heatmap = heatmap.numpy()

    emotion_name = (
        model.layers[-1].name  # not reliable; better to pass label names
    )
    # We'll return the class index and let caller map to label
    return heatmap, target_class


def overlay_heatmap(
    face_bgr: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """
    Overlay a Grad-CAM heatmap on a BGR face image.

    Args:
      face_bgr: original face crop (BGR format, any size).
      heatmap: 2D array (H, W) with values in [0, 1].
      alpha: transparency of heatmap overlay (0 = invisible, 1 = full).
      colormap: OpenCV colormap for the heatmap.

    Returns:
      BGR image with heatmap overlaid.
    """
    # Resize heatmap to match face image size
    heatmap_resized = cv2.resize(heatmap, (face_bgr.shape[1], face_bgr.shape[0]))

    # Convert to 8-bit
    heatmap_uint8 = (heatmap_resized * 255).astype(np.uint8)

    # Apply colormap (produces BGR)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, colormap)

    # Overlay
    overlay = cv2.addWeighted(face_bgr, 1.0 - alpha, heatmap_color, alpha, 0)

    return overlay


def explain_prediction(
    model: tf.keras.Model,
    face_bgr: np.ndarray,
    emotion_labels: list[str],
    target_class: int | None = None,
) -> dict:
    """
    Full Grad-CAM explanation for a face image.

    Args:
      model: trained model.
      face_bgr: BGR face crop (any size).
      emotion_labels: list of 7 emotion names.
      target_class: optional class to explain.

    Returns:
      dict with heatmap, overlay, predicted class, and explanation text.
    """
    # Preprocess face for model (resize to 48x48, grayscale, normalize)
    face_resized = cv2.resize(face_bgr, (48, 48))
    if len(face_resized.shape) == 3:
        face_gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
    else:
        face_gray = face_resized
    face_norm = face_gray.astype(np.float32) / 255.0
    face_input = np.expand_dims(face_norm, axis=(0, -1))

    # Predict
    probs = model.predict(face_input, verbose=0)[0]
    pred_class = int(np.argmax(probs))
    pred_emotion = emotion_labels[pred_class]
    pred_confidence = float(probs[pred_class])

    # Grad-CAM
    heatmap, explained_class = compute_gradcam(model, face_input, target_class)
    overlay = overlay_heatmap(face_bgr, heatmap)

    return {
        "predicted_class": pred_class,
        "predicted_emotion": pred_emotion,
        "confidence": pred_confidence,
        "all_probs": {emotion_labels[i]: float(probs[i])
                      for i in range(len(probs))},
        "heatmap": heatmap,
        "overlay": overlay,
        "explained_class": explained_class,
        "explanation": (
            f"The model predicts **{pred_emotion}** with {pred_confidence*100:.1f}% confidence. "
            f"Grad-CAM highlights the regions that most influenced this prediction. "
            f"Brighter areas (red/yellow) contributed more to the decision. "
            f"This shows model attention, not biological causation."
        ),
    }


class GradCAMExplainer:
    """
    Wrapper class for Grad-CAM, providing a convenient interface for Gradio app.

    Usage:
        explainer = GradCAMExplainer(model)
        result = explainer.explain(face_input, face_crop, EMOTION_LABELS)
    """

    def __init__(self, model: tf.keras.Model):
        self.model = model

    def explain(
        self,
        face_input: np.ndarray,
        face_bgr: np.ndarray,
        emotion_labels: list[str],
        target_class: int | None = None,
    ) -> dict:
        """Run Grad-CAM explanation on a preprocessed face input and BGR crop."""
        return explain_prediction(
            self.model, face_bgr, emotion_labels, target_class
        )


if __name__ == "__main__":
    # Quick test: load model and run Grad-CAM on a sample
    try:
        from src.model import EMOTION_LABELS
        from src.preprocess import load_preprocessed
    except ModuleNotFoundError:
        from model import EMOTION_LABELS
        from preprocess import load_preprocessed
    from tensorflow.keras.models import load_model

    print("Loading model and data for Grad-CAM test...")
    model = load_model("models/emotion_model.keras")
    data = load_preprocessed()

    # Take first test sample
    idx = 0
    X_test = data["X_test"]
    y_test = data["y_test"]

    # Convert from (1, 48, 48, 1) to BGR for display
    face_norm = X_test[idx, :, :, 0]  # (48, 48)
    face_uint8 = (face_norm * 255).astype(np.uint8)
    face_bgr = cv2.cvtColor(face_uint8, cv2.COLOR_GRAY2BGR)

    result = explain_prediction(model, face_bgr, EMOTION_LABELS)

    print(f"\nGrad-CAM test on sample {idx}:")
    print(f"  True label:     {EMOTION_LABELS[y_test[idx]]}")
    print(f"  Predicted:      {result['predicted_emotion']} ({result['confidence']*100:.1f}%)")
    print(f"  Heatmap shape:  {result['heatmap'].shape}")
    print(f"  Heatmap range:  [{result['heatmap'].min():.3f}, {result['heatmap'].max():.3f}]")

    # Save overlay and heatmap
    cv2.imwrite("results/gradcam_overlay.png", result["overlay"])
    cv2.imwrite("results/gradcam_heatmap.png",
                (result["heatmap"] * 255).astype(np.uint8))
    print(f"\nSaved: results/gradcam_overlay.png")
    print(f"Saved: results/gradcam_heatmap.png")
