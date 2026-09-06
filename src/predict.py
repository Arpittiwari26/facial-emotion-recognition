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
    allow_fallback: bool = True,
) -> list[tuple[int, int, int, int]]:
    """Detect faces with OpenCV Haar cascade. Returns list of (x, y, w, h)."""
    if image is None or image.size == 0:
        return []

    h_img, w_img = image.shape[:2]

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    try:
        if not cascade_path:
            cascade_path = find_cascade()

        cascade_cls = getattr(cv2, "CascadeClassifier", None)
        if cascade_cls is not None and Path(cascade_path).exists():
            cascade = cascade_cls(str(cascade_path))
            if not cascade.empty():
                # Pass 1: Standard detection
                faces = cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
                )
                if len(faces) > 0:
                    return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]

                # Pass 2: Looser detection on equalized image
                gray_eq = cv2.equalizeHist(gray)
                faces = cascade.detectMultiScale(
                    gray_eq, scaleFactor=1.05, minNeighbors=3, minSize=(20, 20)
                )
                if len(faces) > 0:
                    return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]
    except Exception as e:
        print(f"Warning in detect_faces: {e}")

    # Fallback to full image if no bounding box found so prediction always runs
    if allow_fallback:
        return [(0, 0, w_img, h_img)]

    return []


def preprocess_face(face_bgr: np.ndarray) -> np.ndarray:
    """Crop, resize 48x48, grayscale, normalize, add dims -> (1,48,48,1)."""
    if face_bgr is None or face_bgr.size == 0:
        return np.zeros((1, IMG_SIZE, IMG_SIZE, 1), dtype=np.float32)

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
