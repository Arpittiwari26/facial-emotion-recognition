"""
app.py

Gradio web application for Facial Emotion Recognition.
Provides continuous browser-webcam streaming, image upload,
Grad-CAM explainability, and dataset metrics.

Compatible with Hugging Face Spaces.
"""

from __future__ import annotations

import io
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
import gradio as gr
from PIL import Image

# Ensure src/ is on sys.path
_SCRIPT_DIR = Path(__file__).parent
_SRC_DIR = _SCRIPT_DIR / "src"
if str(_SRC_DIR.resolve()) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR.resolve()))

try:
    from src.model import EMOTION_LABELS
    from src.predict import detect_faces, preprocess_face, find_cascade
    from src.gradcam import explain_prediction
except ModuleNotFoundError:
    from model import EMOTION_LABELS
    from predict import detect_faces, preprocess_face, find_cascade
    from gradcam import explain_prediction

# ── Paths & Model Loading ────────────────────────────────────────────
MODEL_PATH = Path("models/emotion_model.keras")
CASCADE_PATH = find_cascade()

model: tf.keras.Model | None = None
if MODEL_PATH.exists():
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print(f"[app] Loaded model from {MODEL_PATH}")
    except Exception as e:
        print(f"[app] Failed to load model: {e}")
else:
    print(f"[app] Warning: Model file not found at {MODEL_PATH}")


# ── Temporal Smoothing Class ──────────────────────────────────────────
class EmotionSmoothie:
    """Rolling temporal smoother for live frame predictions."""

    def __init__(self, window_size: int = 5):
        self.window_size = window_size
        self.history: list[np.ndarray] = []

    def update(self, probs: np.ndarray) -> np.ndarray:
        self.history.append(probs)
        if len(self.history) > self.window_size:
            self.history.pop(0)
        return np.mean(self.history, axis=0)

    def reset(self):
        self.history.clear()


# Global smoothie instance & frame timer for live streaming
global_smoother = EmotionSmoothie(window_size=5)
last_frame_time = [time.time()]


# ── 1. Live Stream Handler ────────────────────────────────────────────
def predict_live_stream(frame_rgb: np.ndarray | None):
    """
    Process continuous browser webcam frames from Gradio streaming input.
    Inputs: frame_rgb (numpy array from browser webcam).
    Returns: (annotated_frame_rgb, emotion_probabilities_dict, status_fps_str)
    """
    if frame_rgb is None or frame_rgb.size == 0:
        return None, {"No face detected": 1.0}, "0.0 FPS | Waiting for browser camera..."

    if model is None:
        return frame_rgb, {"Model missing": 1.0}, "Error: Model file not loaded."

    now = time.time()
    dt = now - last_frame_time[0]
    last_frame_time[0] = now
    fps = (1.0 / dt) if (dt > 0 and dt < 2.0) else 0.0

    # Convert RGB frame from Gradio to BGR for OpenCV
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    annotated_bgr = frame_bgr.copy()

    faces = detect_faces(frame_bgr, CASCADE_PATH, allow_fallback=False)

    if len(faces) == 0:
        # Overlay clear status banner
        cv2.putText(
            annotated_bgr, "No face detected", (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2
        )
        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
        return annotated_rgb, {"No face detected": 1.0}, f"FPS: {fps:.1f} | Status: No face detected"

    summary_probs: dict[str, float] = {}

    for i, (x, y, w, h) in enumerate(faces):
        face_crop = frame_bgr[y:y+h, x:x+w]
        if face_crop.size == 0:
            continue

        face_input = preprocess_face(face_crop)
        raw_probs = model.predict(face_input, verbose=0)[0]

        # Apply temporal smoothing across recent frames
        smoothed_probs = global_smoother.update(raw_probs)
        pred_idx = int(np.argmax(smoothed_probs))
        pred_emotion = EMOTION_LABELS[pred_idx]
        confidence = float(smoothed_probs[pred_idx])

        # Draw bounding box
        color = (0, 255, 0)
        cv2.rectangle(annotated_bgr, (x, y), (x+w, y+h), color, 2)
        label_text = f"{pred_emotion} {confidence*100:.0f}%"

        # Label box background
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        y_label = max(y - 8, th + 8)
        cv2.rectangle(
            annotated_bgr,
            (x, y_label - th - 4), (x + tw + 6, y_label + 4),
            color, -1
        )
        cv2.putText(
            annotated_bgr, label_text, (x + 3, y_label),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2
        )

        for j, em_name in enumerate(EMOTION_LABELS):
            summary_probs[em_name] = float(smoothed_probs[j])

    # Overlay FPS counter at top right
    cv2.putText(
        annotated_bgr, f"FPS: {fps:.1f}", (annotated_bgr.shape[1] - 120, 35),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
    )

    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
    status_str = f"FPS: {fps:.1f} | Detected {len(faces)} face(s)"
    return annotated_rgb, summary_probs, status_str


# ── 2. Upload Image Handler ───────────────────────────────────────────
def predict_uploaded_image(image_rgb: np.ndarray | None):
    """
    Process single uploaded image.
    Returns: (annotated_image_rgb, emotion_probabilities_dict, status_str)
    """
    if image_rgb is None or image_rgb.size == 0:
        return None, {}, "Please upload an image."

    if model is None:
        return image_rgb, {}, "Error: Model file not loaded."

    frame_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    annotated_bgr = frame_bgr.copy()

    faces = detect_faces(frame_bgr, CASCADE_PATH, allow_fallback=True)

    summary_probs: dict[str, float] = {}

    for i, (x, y, w, h) in enumerate(faces):
        face_crop = frame_bgr[y:y+h, x:x+w]
        if face_crop.size == 0:
            continue

        face_input = preprocess_face(face_crop)
        probs = model.predict(face_input, verbose=0)[0]
        pred_idx = int(np.argmax(probs))
        pred_emotion = EMOTION_LABELS[pred_idx]
        confidence = float(probs[pred_idx])

        color = (0, 255, 0)
        cv2.rectangle(annotated_bgr, (x, y), (x+w, y+h), color, 2)
        label_text = f"{pred_emotion} {confidence*100:.1f}%"

        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        y_label = max(y - 8, th + 8)
        cv2.rectangle(
            annotated_bgr,
            (x, y_label - th - 4), (x + tw + 6, y_label + 4),
            color, -1
        )
        cv2.putText(
            annotated_bgr, label_text, (x + 3, y_label),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2
        )

        for j, em_name in enumerate(EMOTION_LABELS):
            summary_probs[em_name] = float(probs[j])

    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
    status_str = f"Analysis complete. Detected {len(faces)} face region(s)."
    return annotated_rgb, summary_probs, status_str


# ── 3. Grad-CAM Handler ───────────────────────────────────────────────
def explain_gradcam_image(image_rgb: np.ndarray | None, target_emotion: str):
    """
    Generate Grad-CAM heatmap explainability visualization.
    Returns: (overlay_image_rgb, raw_heatmap_rgb, explanation_md)
    """
    if image_rgb is None or image_rgb.size == 0:
        return None, None, "Please upload an image for Grad-CAM explanation."

    if model is None:
        return None, None, "Error: Model file not loaded."

    frame_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    faces = detect_faces(frame_bgr, CASCADE_PATH, allow_fallback=True)

    if len(faces) == 0:
        return None, None, "No face detected in the image."

    x, y, w, h = faces[0]
    face_crop = frame_bgr[y:y+h, x:x+w]

    target_class = None
    if target_emotion in EMOTION_LABELS:
        target_class = EMOTION_LABELS.index(target_emotion)

    res = explain_prediction(model, face_crop, EMOTION_LABELS, target_class=target_class)

    overlay_bgr = res["overlay"]
    overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)

    heatmap = res["heatmap"]
    heatmap_color_bgr = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_color_bgr, cv2.COLOR_BGR2RGB)

    explanation_text = res["explanation"]
    return overlay_rgb, heatmap_rgb, explanation_text


# ── Build Gradio Interface ───────────────────────────────────────────
with gr.Blocks(title="Facial Emotion Recognition") as demo:
    gr.Markdown("""
    # 😊 Facial Emotion Recognition System
    Real-time browser webcam facial emotion recognition trained on the **FER-2013** dataset (35,887 grayscale 48×48 face images).
    """)

    with gr.Tabs():
        # ── TAB 1: LIVE BROWSER WEBCAM ──────────────────────────────────
        with gr.TabItem("🎥 Live Expression"):
            gr.Markdown("""
            ### 🎥 Live Browser Webcam Feed
            Click the camera component below to turn on your browser webcam.
            The frame stream will automatically detect faces, run CNN inference, apply temporal smoothing, and display bounding boxes with real-time emotion predictions!
            """)

            with gr.Row():
                webcam_input = gr.Image(
                    sources=["webcam"],
                    streaming=True,
                    type="numpy",
                    label="Browser Webcam Input",
                )
                live_output_image = gr.Image(
                    label="Live Annotated Stream",
                    type="numpy",
                )

            with gr.Row():
                live_label_output = gr.Label(
                    num_top_classes=7,
                    label="Live Emotion Probabilities",
                )
                live_status_output = gr.Textbox(
                    label="Stream Status / FPS",
                    interactive=False,
                )

            webcam_input.stream(
                fn=predict_live_stream,
                inputs=[webcam_input],
                outputs=[live_output_image, live_label_output, live_status_output],
                stream_every=0.04,
            )

        # ── TAB 2: UPLOAD IMAGE ─────────────────────────────────────────
        with gr.TabItem("📸 Upload Image"):
            gr.Markdown("### 📸 Single Image Emotion Prediction")
            with gr.Row():
                upload_input = gr.Image(
                    sources=["upload"],
                    type="numpy",
                    label="Upload Image",
                )
                upload_output_image = gr.Image(
                    label="Annotated Result",
                    type="numpy",
                )

            with gr.Row():
                upload_label_output = gr.Label(
                    num_top_classes=7,
                    label="Emotion Probabilities",
                )
                upload_status_output = gr.Textbox(
                    label="Status",
                    interactive=False,
                )

            upload_button = gr.Button("Predict Emotion", variant="primary")
            upload_button.click(
                fn=predict_uploaded_image,
                inputs=[upload_input],
                outputs=[upload_output_image, upload_label_output, upload_status_output],
            )

        # ── TAB 3: GRAD-CAM ─────────────────────────────────────────────
        with gr.TabItem("🔍 Grad-CAM"):
            gr.Markdown("### 🔍 Grad-CAM Explainability Visualization")
            gr.Markdown("Grad-CAM highlights the facial regions (mouth, eyes, eyebrows) that most influenced the CNN model's prediction.")

            with gr.Row():
                gradcam_input = gr.Image(
                    sources=["upload"],
                    type="numpy",
                    label="Upload Face Image",
                )
                gradcam_target_dropdown = gr.Dropdown(
                    choices=["Auto (Predicted)"] + EMOTION_LABELS,
                    value="Auto (Predicted)",
                    label="Target Emotion Class to Explain",
                )

            with gr.Row():
                gradcam_overlay_output = gr.Image(
                    label="Grad-CAM Overlay Heatmap",
                    type="numpy",
                )
                gradcam_raw_output = gr.Image(
                    label="Raw Attention Heatmap",
                    type="numpy",
                )

            gradcam_explanation_text = gr.Markdown()
            gradcam_button = gr.Button("Generate Grad-CAM Explanation", variant="primary")
            gradcam_button.click(
                fn=explain_gradcam_image,
                inputs=[gradcam_input, gradcam_target_dropdown],
                outputs=[gradcam_overlay_output, gradcam_raw_output, gradcam_explanation_text],
            )

        # ── TAB 4: METRICS & SYSTEM INFO ────────────────────────────────
        with gr.TabItem("📊 Model Metrics & System Info"):
            gr.Markdown("""
            ### 📊 Model Architecture & Performance Details

            - **Dataset**: FER-2013 (Kaggle) — 35,887 grayscale 48×48 pixel face images
            - **Emotion Classes (7)**: `0: Angry`, `1: Disgust`, `2: Fear`, `3: Happy`, `4: Sad`, `5: Surprise`, `6: Neutral`
            - **CNN Architecture**: 3-Block VGG-style Conv2D (32→64→128) + Flatten + Dense(256) + Dense(7, Softmax)
            - **Test Accuracy**: **52.94%** across 3,589 real test set images
            - **Pre-processing**: Face crop → Grayscale conversion → 48×48 Resize → Pixel Normalization (`[0, 1]`) → Batch Dimension `(1, 48, 48, 1)`
            - **Deployment**: Hugging Face Spaces + Gradio (Browser Webcam Streaming)
            """)

# ── Main Entry Point ──────────────────────────────────────────────────
if __name__ == "__main__":
    demo.launch(share=True)
