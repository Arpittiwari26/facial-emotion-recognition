"""
src/predict.py

Predict emotion on a single image file.

Usage:
  python src/predict.py path/to/image.jpg
  python src/predict.py path/to/image.jpg --model models/emotion_model.keras

For each detected face, prints:
  - Predicted emotion (top-1)
  - Confidence percentage
  - Bar chart of all 7 emotion probabilities
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
try:
    from src.model import EMOTION_LABELS
except ModuleNotFoundError:
    from model import EMOTION_LABELS

from tensorflow.keras.models import load_model

IMG_SIZE = 48


def find_cascade() -> str:
    """Locate the OpenCV Haar cascade XML for face detection."""
    candidates = [
        str(Path("models/haarcascade_frontalface_default.xml").resolve()),
        str(Path("src/haarcascade_frontalface_default.xml").resolve()),
    ]
    if hasattr(cv2, "data") and hasattr(cv2.data, "haarcascades"):
        candidates.append(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    for p in candidates:
        if p and Path(p).exists():
            return p
    return candidates[0]


def detect_faces(
    image: np.ndarray,
    cascade_path: str = "",
    margin: float = 0.1,
    allow_fallback: bool = False,
) -> list[tuple[int, int, int, int]]:
    """
    Detect faces in an image (BGR or grayscale). Returns list of (x, y, w, h).
    Uses multi-stage cascade detection and optional padding margin.
    """
    if image is None or image.size == 0:
        return []

    h_img, w_img = image.shape[:2]

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    faces = []
    try:
        if not cascade_path:
            cascade_path = find_cascade()

        cascade_cls = getattr(cv2, "CascadeClassifier", None)
        if cascade_cls is not None and Path(cascade_path).exists():
            cascade = cascade_cls(str(cascade_path))
            if not cascade.empty():
                # Equalize grayscale for robust face detection in varying lighting
                gray_eq = cv2.equalizeHist(gray)

                # Pass 1: Standard detection on equalized grayscale
                detected = cascade.detectMultiScale(
                    gray_eq, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
                )
                # Pass 2: Looser detection on raw grayscale if Pass 1 found nothing
                if len(detected) == 0:
                    detected = cascade.detectMultiScale(
                        gray, scaleFactor=1.05, minNeighbors=3, minSize=(30, 30)
                    )

                if len(detected) > 0:
                    for (x, y, w, h) in detected:
                        if margin > 0:
                            pad_w = int(w * margin)
                            pad_h = int(h * margin)
                            x_new = max(0, x - pad_w)
                            y_new = max(0, y - pad_h)
                            w_new = min(w_img - x_new, w + 2 * pad_w)
                            h_new = min(h_img - y_new, h + 2 * pad_h)
                            faces.append((int(x_new), int(y_new), int(w_new), int(h_new)))
                        else:
                            faces.append((int(x), int(y), int(w), int(h)))
                    return faces
    except Exception as e:
        print(f"Warning in detect_faces: {e}")

    # Fallback: if allow_fallback is True OR image is already a small cropped face (<= 180x180)
    if allow_fallback or (w_img <= 180 and h_img <= 180):
        return [(0, 0, w_img, h_img)]

    return []


def preprocess_face(face_bgr: np.ndarray) -> np.ndarray:
    """Resize to 48x48, grayscale, normalize to [0,1], add batch+channel dims."""
    face_resized = cv2.resize(face_bgr, (IMG_SIZE, IMG_SIZE))
    if len(face_resized.shape) == 3:
        face_gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
    else:
        face_gray = face_resized
    face_norm = face_gray.astype(np.float32) / 255.0
    return np.expand_dims(face_norm, axis=(0, -1))


def predict_image(
    image_path: str | Path,
    model: tf.keras.Model,
    cascade_path: str,
) -> dict:
    """Detect faces in image, run model, return structured results."""
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detect_faces(gray, cascade_path)

    if len(faces) == 0:
        print("\n[predict] No face detected.", file=sys.stderr)
        print("[predict] Trying whole-image as fallback...", file=sys.stderr)
        h, w = gray.shape[:2]
        faces = [(0, 0, w, h)]

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
            "face_index": i + 1,
            "bbox": (x, y, w, h),
            "emotion": pred_emotion,
            "confidence": confidence,
            "all_probs": {EMOTION_LABELS[j]: float(probs[j])
                          for j in range(len(probs))},
        })
    return {"image_path": str(image_path), "faces": results}


def print_results(results: dict):
    """Pretty-print prediction results."""
    print(f"\n{'='*55}")
    print(f"  Facial Emotion Recognition — Image Prediction")
    print(f"{'='*55}")
    print(f"  Image: {results['image_path']}")
    print(f"  Faces detected: {len(results['faces'])}")
    print(f"{'='*55}")

    for face in results["faces"]:
        print(f"\n  Face #{face['face_index']}")
        print(f"  BBox: {face['bbox']}")
        print(f"\n  >>> Predicted emotion: {face['emotion']}")
        print(f"  >>> Confidence:       {face['confidence']*100:.2f}%")
        print(f"\n  All emotion probabilities:")
        print(f"  {'Emotion':<12} {'Probability':>12}  {'Visualization'}")
        print(f"  {'-'*10}  {'-'*12}  {'-'*25}")
        for emotion, prob in sorted(face["all_probs"].items(),
                                     key=lambda x: -x[1]):
            bar = "█" * int(prob * 35)
            print(f"  {emotion:<12} {prob*100:>11.2f}%  {bar}")


def cli():
    parser = argparse.ArgumentParser(
        description="Predict emotion from an image file"
    )
    parser.add_argument("image", type=str, help="Path to input image")
    parser.add_argument("--model", type=str,
                        default="models/emotion_model.keras",
                        help="Path to trained .keras model")
    parser.add_argument("--cascade", type=str, default=None,
                        help="Path to Haar cascade XML")
    args = parser.parse_args()

    print("=" * 60)
    print("  Facial Emotion Recognition — Prediction")
    print("=" * 60)

    model = load_model(args.model)
    cascade = args.cascade or find_cascade()

    print(f"\nModel: {args.model}")
    print(f"Image: {args.image}")
    print(f"Cascade: {Path(cascade).name}\n")

    try:
        results = predict_image(args.image, model, cascade)
        print_results(results)
    except FileNotFoundError as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    cli()
