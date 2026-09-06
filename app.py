"""
app.py

Streamlit web application for the Facial Emotion Recognition system.

Features:
  - Upload an image → face detection → emotion prediction + Grad-CAM
  - Webcam capture (browser-based single photo)
  - Model performance metrics (confusion matrix, training plots, per-class metrics)
  - Interactive Grad-CAM explainability

PRIMARY real-time demo is still src/webcam.py (OpenCV native window).
This Streamlit app complements it for upload-based testing, metrics viewing,
and Grad-CAM visualization.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

# Ensure src/ is on the path so Streamlit can import src/* modules
# when running from the project root (streamlit run app.py)
_SCRIPT_DIR = Path(__file__).parent
_SRC_DIR = _SCRIPT_DIR / "src"
if str(_SRC_DIR.resolve()) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR.resolve()))

import cv2
import numpy as np
import streamlit as st
import tensorflow as tf

from model import EMOTION_LABELS, load_model
from predict import detect_faces, preprocess_face, find_cascade
from gradcam import GradCAMExplainer

# ── Paths ────────────────────────────────────────────────────────────
MODEL_PATH = Path("models/emotion_model.keras")
RESULTS_DIR = Path("results")
CASCADE_PATH = find_cascade()

st.set_page_config(
    page_title="Facial Emotion Recognition",
    page_icon="😊",
    layout="wide",
)


# ── Sidebar ──────────────────────────────────────────────────────────
with st.sidebar:
    st.title("😊 Facial Emotion Recognition")
    st.markdown("""
    **Real-time facial emotion recognition** using a CNN trained on
    the FER-2013 dataset (35,887 grayscale 48×48 face images).

    Built with TensorFlow/Keras, OpenCV, and Streamlit.
    """)

    st.markdown("---")

    # Model status
    st.subheader("Model Status")
    if MODEL_PATH.exists():
        st.success("✅ Model found")
        try:
            model = tf.keras.models.load_model(MODEL_PATH)
            st.write(f"**Parameters:** {model.count_params():,}")
            st.write(f"**Input shape:** {model.input_shape}")
            st.write(f"**Output shape:** {model.output_shape}")
        except Exception as e:
            st.error(f"Failed to load: {e}")
    else:
        st.warning("⚠️ Model not found")
        st.info("Train first: `python src/train.py`")

    st.markdown("---")
    st.markdown("""
    **7 Emotions:**
    | # | Emotion |
    |---|---------|
    | 0 | 😠 Angry |
    | 1 | 🤢 Disgust |
    | 2 | 😨 Fear |
    | 3 | 😄 Happy |
    | 4 | 😢 Sad |
    | 5 | 😲 Surprise |
    | 6 | 😐 Neutral |
    """)

    st.markdown("---")
    st.markdown("""
    **Quick Commands:**
    ```bash
    python src/train.py          # Train model
    python src/evaluate.py       # Evaluate
    python src/predict.py img.jpg  # Single image
    python src/webcam.py         # REAL-TIME webcam ← main demo
    streamlit run app.py         # This app
    ```
    """)


# ── Load model (lazy, for upload/webcam tabs) ───────────────────────
def get_model():
    """Load model on demand. Returns (model, error_msg)."""
    if not MODEL_PATH.exists():
        return None, "Model not found. Train with `python src/train.py`."
    try:
        m = tf.keras.models.load_model(MODEL_PATH)
        return m, None
    except Exception as e:
        return None, f"Failed to load model: {e}"


# ── Tabs ─────────────────────────────────────────────────────────────
tab_upload, tab_webcam, tab_metrics, tab_gradcam = st.tabs([
    "📸 Upload Image",
    "🎥 Webcam Photo",
    "📊 Model Metrics",
    "🔍 Grad-CAM",
])


# ═════════════════════════════════════════════════════════════════════
# TAB 1: UPLOAD IMAGE
# ═════════════════════════════════════════════════════════════════════

with tab_upload:
    st.header("Upload an Image for Emotion Prediction")
    st.markdown("""
    The image is processed as follows:
    1. OpenCV detects faces using a Haar cascade
    2. Each detected face is cropped, resized to 48×48, converted to grayscale
    3. The CNN predicts probabilities for all 7 emotions
    4. Results are displayed with bounding boxes, emotion labels, and
       probability distributions
    """)

    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
    )

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if image is None:
            st.error("Could not decode image. Try a different file.")
        else:
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Uploaded Image")
                st.image(image, channels="BGR", use_container_width=True)
                st.write(f"Size: {image.shape[1]}×{image.shape[0]} pixels")

            with col2:
                st.subheader("Prediction")
                model, err = get_model()
                if err:
                    st.error(err)
                else:
                    try:
                        faces = detect_faces(image, CASCADE_PATH)

                        if len(faces) == 0:
                            st.warning("No face detected in the image.")
                            st.info("Tips: Ensure the face is clearly visible, "
                                    "well-lit, and facing forward. "
                                    "The Haar cascade works best on frontal faces.")
                        else:
                            annotated = image.copy()
                            results = []

                            for i, (x, y, w, h) in enumerate(faces):
                                face_crop = image[y:y+h, x:x+w]
                                if face_crop.size == 0:
                                    continue

                                face_input = preprocess_face(face_crop)
                                probs = model.predict(face_input, verbose=0)[0]
                                pred_idx = int(np.argmax(probs))
                                pred_emotion = EMOTION_LABELS[pred_idx]
                                confidence = float(probs[pred_idx])

                                results.append({
                                    "bbox": (x, y, w, h),
                                    "emotion": pred_emotion,
                                    "confidence": confidence,
                                    "probs": {EMOTION_LABELS[j]: float(probs[j])
                                              for j in range(len(probs))},
                                })

                                # Draw on annotated image
                                color = (0, 255, 0)
                                cv2.rectangle(
                                    annotated, (x, y), (x+w, y+h), color, 2
                                )
                                label = f"{pred_emotion} {confidence*100:.1f}%"
                                (tw, th), _ = cv2.getTextSize(
                                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                                )
                                cv2.rectangle(
                                    annotated,
                                    (x, y+h+8), (x+tw+8, y+h+8+th+4),
                                    color, -1,
                                )
                                cv2.putText(
                                    annotated, label, (x+4, y+h+8+th),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                    (255, 255, 255), 2,
                                )

                                # Mini probability bars
                                bar_x = x
                                bar_y = y + h + 30
                                bar_w = min(w, 150)
                                bar_h = 4
                                sorted_pairs = sorted(
                                    enumerate(probs), key=lambda x: -x[1]
                                )[:3]
                                for j, (idx, prob) in enumerate(sorted_pairs):
                                    bar_color = (0, 255, 0) if idx == pred_idx else (100, 100, 100)
                                    cv2.rectangle(
                                        annotated,
                                        (bar_x, bar_y + j*10),
                                        (bar_x + int(bar_w * prob),
                                         bar_y + j*10 + bar_h),
                                        bar_color, -1,
                                    )
                                    cv2.putText(
                                        annotated,
                                        f"{EMOTION_LABELS[idx]} {prob*100:.0f}%",
                                        (bar_x + int(bar_w*prob) + 3,
                                         bar_y + j*10 + bar_h - 2),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                                        bar_color, 1,
                                    )

                            st.image(
                                annotated, channels="BGR",
                                use_container_width=True,
                            )

                            for i, res in enumerate(results):
                                with st.expander(
                                    f"Face #{i+1}: {res['emotion']} "
                                    f"({res['confidence']*100:.1f}%)"
                                ):
                                    st.write(f"**Bounding box:** {res['bbox']}")
                                    st.write(f"**Predicted emotion:** "
                                             f"{res['emotion']}")
                                    st.write(f"**Confidence:** "
                                             f"{res['confidence']*100:.2f}%")
                                    st.write("**All emotion probabilities:**")
                                    sorted_probs = sorted(
                                        res["probs"].items(),
                                        key=lambda x: -x[1],
                                    )
                                    for em, prob in sorted_probs:
                                        st.write(f"  - {em}: {prob*100:.2f}%")

                    except Exception as e:
                        st.error(f"Prediction error: {e}")


# ═════════════════════════════════════════════════════════════════════
# TAB 2: WEBCAM PHOTO
# ═════════════════════════════════════════════════════════════════════

with tab_webcam:
    st.header("Webcam Photo Capture")
    st.markdown("""
    Take a single photo with your browser's webcam.

    **For continuous real-time video** (the main demo of this project),
    run the dedicated OpenCV application:

    ```bash
    python src/webcam.py
    ```

    That gives you a live window with bounding boxes, emotion labels,
    FPS counter, and temporal smoothing — continuously.
    """)

    webcam_photo = st.camera_input("Take a photo with your webcam")

    if webcam_photo is not None:
        file_bytes = np.asarray(bytearray(webcam_photo.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if image is None:
            st.error("Could not decode webcam photo.")
        else:
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Captured Photo")
                st.image(image, channels="BGR", use_container_width=True)

            with col2:
                st.subheader("Prediction")
                model, err = get_model()
                if err:
                    st.error(err)
                else:
                    try:
                        faces = detect_faces(image, CASCADE_PATH)

                        if len(faces) == 0:
                            st.warning("No face detected in the photo.")
                        else:
                            annotated = image.copy()
                            for i, (x, y, w, h) in enumerate(faces):
                                face_crop = image[y:y+h, x:x+w]
                                if face_crop.size == 0:
                                    continue

                                face_input = preprocess_face(face_crop)
                                probs = model.predict(face_input, verbose=0)[0]
                                pred_idx = int(np.argmax(probs))
                                pred_emotion = EMOTION_LABELS[pred_idx]
                                confidence = float(probs[pred_idx])

                                color = (0, 255, 0)
                                cv2.rectangle(
                                    annotated, (x, y), (x+w, y+h), color, 2
                                )
                                label = f"{pred_emotion} {confidence*100:.1f}%"
                                cv2.putText(
                                    annotated, label, (x, y-8),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                    color, 2,
                                )

                            st.image(
                                annotated, channels="BGR",
                                use_container_width=True,
                            )

                            for i, (x, y, w, h) in enumerate(faces):
                                face_crop = image[y:y+h, x:x+w]
                                if face_crop.size == 0:
                                    continue
                                face_input = preprocess_face(face_crop)
                                probs = model.predict(face_input, verbose=0)[0]
                                pred_idx = int(np.argmax(probs))
                                st.success(
                                    f"Face #{i+1}: **{EMOTION_LABELS[pred_idx]}** "
                                    f"({probs[pred_idx]*100:.1f}%)"
                                )
                    except Exception as e:
                        st.error(f"Prediction error: {e}")


# ═════════════════════════════════════════════════════════════════════
# TAB 3: MODEL METRICS
# ═════════════════════════════════════════════════════════════════════

with tab_metrics:
    st.header("Model Performance Metrics")

    # Confusion matrix
    cm_path = RESULTS_DIR / "confusion_matrix.png"
    if cm_path.exists():
        st.subheader("Confusion Matrix")
        st.image(
            str(cm_path),
            caption="Normalized confusion matrix on FER-2013 test set "
                    "(rows = true emotion, columns = predicted emotion)",
            use_container_width=True,
        )
        st.markdown("""
        **How to read:**
        - Each row sums to 1.0 (all samples of that true emotion).
        - Diagonal = correct predictions.
        - Off-diagonal = misclassifications. High values show commonly confused emotions.
        - Example: if "Fear → Surprise" has a high value, the model often mistakes fear for surprise.
        """)
    else:
        st.info("Confusion matrix not found. Run: `python src/evaluate.py`")

    # Training plots
    hist_path = RESULTS_DIR / "training_history.png"
    if hist_path.exists():
        st.subheader("Training History")
        st.image(
            str(hist_path),
            caption="Training vs validation accuracy and loss over epochs",
            use_container_width=True,
        )
        st.markdown("""
        **Identifying model behavior:**

        | Pattern | Diagnosis |
        |---------|-----------|
        | Train & val accuracy rise together, plateau | ✅ Good convergence |
        | Train accuracy rises, val accuracy plateaus/drops | ⚠️ Overfitting |
        | Both train & val accuracy stay low | 🔄 Underfitting |
        | Train loss decreases, val loss increases after some epoch | ⚠️ Overfitting |

        EarlyStopping should stop training when overfitting begins, saving
        the best checkpoint before validation performance degrades.
        """)
    else:
        st.info("Training plots not found. Run: `python src/train.py`")

    # Numeric metrics from JSON
    metrics_path = RESULTS_DIR / "metrics.json"
    if metrics_path.exists():
        import json
        with open(metrics_path) as f:
            metrics = json.load(f)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric(
            "Test Accuracy",
            f"{metrics.get('test_accuracy', 0)*100:.2f}%",
        )
        col2.metric("Test Loss", f"{metrics.get('test_loss', 0):.4f}")
        col3.metric("Macro F1", f"{metrics.get('macro_f1', 0):.4f}")
        col4.metric("Weighted F1", f"{metrics.get('weighted_f1', 0):.4f}")

        st.markdown("---")
        st.subheader("Per-Class Performance")

        if "per_class" in metrics:
            import pandas as pd
            rows = []
            for i, name in enumerate(EMOTION_LABELS):
                pc = metrics["per_class"].get(str(i), {})
                rows.append({
                    "Emotion": name,
                    "Precision": f"{pc.get('precision', 0):.4f}",
                    "Recall": f"{pc.get('recall', 0):.4f}",
                    "F1": f"{pc.get('f1', 0):.4f}",
                    "Support": pc.get('support', 0),
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
    else:
        st.info("Metrics JSON not found. Run evaluation after training.")

    # Transfer learning comparison placeholder
    st.markdown("---")
    st.subheader("Transfer Learning Comparison (Optional)")
    st.markdown("""
    An optional transfer-learning model (MobileNetV2) is scaffolded in
    `src/model.py` → `build_transfer_model()`.

    To compare: train both models, then collect:

    | Model | Accuracy | Macro F1 | Training Time | Parameters |
    |-------|----------|----------|---------------|------------|
    | Baseline CNN (from scratch) | — | — | — | ~330K |
    | MobileNetV2 (transfer) | — | — | — | ~2.2M |

    **Important:** Transfer learning is NOT automatically better. On
    FER-2013 (48×48 grayscale, 35K images), a from-scratch CNN often
    matches or beats transfer models because ImageNet features are tuned
    for 224×224 RGB natural images, not tiny grayscale faces.
    """)


# ═════════════════════════════════════════════════════════════════════
# TAB 4: GRAD-CAM
# ═════════════════════════════════════════════════════════════════════

with tab_gradcam:
    st.header("Grad-CAM — Model Explainability")

    st.markdown("""
    **Grad-CAM** (Gradient-weighted Class Activation Mapping) produces a
    heatmap showing which regions of the face image most influenced the
    model's prediction.

    ### How It Works

    1. **Forward pass:** Run the image through the model to get the final
       convolutional layer's feature maps and the predictions.
    2. **Backward pass:** Compute the gradient of the predicted class score
       with respect to each feature map.
    3. **Weighting:** Average each gradient map spatially → one weight per
       feature map (how important is this feature map for the prediction?).
    4. **Heatmap:** Weighted sum of feature maps → coarse heatmap.
    5. **ReLU + normalize:** Keep only positive contributions, scale to [0,1].
    6. **Overlay:** Resize heatmap to input size, apply colormap, overlay on
       the original face image.

    ### What the Heatmap Shows

    - **Bright (red/yellow) regions:** areas that strongly contributed to the
      predicted emotion.
    - **Dark (blue/black) regions:** areas that contributed little or
      negatively.

    ### Important Caveat

    > Grad-CAM shows **where the model is looking**, NOT what biologically
    > determines the emotion. A heatmap on the mouth for "Happy" means the
    > model used mouth features for its decision — it does **not** prove
    > that the mouth causes happiness. This is a model interpretability tool,
    > not a causal analysis.
    """)

    gc_file = st.file_uploader(
        "Upload an image for Grad-CAM analysis...",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        key="gradcam_uploader",
    )

    if gc_file is not None:
        file_bytes = np.asarray(bytearray(gc_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if image is None:
            st.error("Could not decode image.")
        else:
            model, err = get_model()
            if err:
                st.error(err)
            else:
                try:
                    faces = detect_faces(image, CASCADE_PATH)

                    if len(faces) == 0:
                        st.warning("No face detected for Grad-CAM.")
                    else:
                        # Show original with face boxes
                        st.subheader("Original Image (Detected Faces)")
                        annotated = image.copy()
                        for (x, y, w, h) in faces:
                            cv2.rectangle(
                                annotated, (x, y), (x+w, y+h), (0, 255, 0), 2
                            )
                        st.image(
                            annotated, channels="BGR",
                            use_container_width=True,
                        )

                        # Grad-CAM for each face
                        for i, (x, y, w, h) in enumerate(faces):
                            face_crop = image[y:y+h, x:x+w]
                            if face_crop.size == 0:
                                continue

                            face_input = preprocess_face(face_crop)

                            explainer = GradCAMExplainer(model)
                            result = explainer.explain(
                                face_input, face_crop, EMOTION_LABELS
                            )

                            with st.expander(
                                f"Face #{i+1} — Grad-CAM Result"
                            ):
                                col_a, col_b = st.columns(2)

                                with col_a:
                                    st.markdown(
                                        f"**Predicted:** {result['predicted_emotion']} "
                                        f"({result['confidence']*100:.1f}%)"
                                    )
                                    st.markdown(result["explanation"])
                                    st.write("**All probabilities:**")
                                    sorted_probs = sorted(
                                        result["all_probabilities"].items(),
                                        key=lambda x: -x[1],
                                    )
                                    for em, prob in sorted_probs:
                                        st.write(f"- {em}: {prob*100:.2f}%")

                                with col_b:
                                    st.markdown("**Grad-CAM Heatmap:**")
                                    st.image(
                                        result["heatmap"],
                                        caption="Heatmap (bright = more important)",
                                        use_container_width=True,
                                    )
                                    st.markdown("**Overlay on Face:**")
                                    st.image(
                                        result["overlay"],
                                        caption="Heatmap overlaid on original face",
                                        use_container_width=True,
                                    )

                                    # Download button
                                    buf = io.BytesIO()
                                    cv2.imencode(".png",
                                                 result["overlay"])[1].tofile(buf)
                                    st.download_button(
                                        label="Download Grad-CAM Overlay",
                                        data=buf.getvalue(),
                                        file_name=f"gradcam_face{i+1}.png",
                                        mime="image/png",
                                    )

                except Exception as e:
                    st.error(f"Grad-CAM error: {e}")


# ── Footer ───────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
**About this project:** A college-level Deep Learning project demonstrating
CNN architecture, training, evaluation, and real-time deployment for facial
emotion recognition.

**Built with:** Python, TensorFlow/Keras, OpenCV, NumPy, Pandas,
Matplotlib, Seaborn, scikit-learn, Streamlit.

**Dataset:** FER-2013 (Kaggle) — 35,887 grayscale 48×48 pixel face images,
7 emotion classes.

**Primary demo:** `python src/webcam.py` (real-time OpenCV window).
""")
