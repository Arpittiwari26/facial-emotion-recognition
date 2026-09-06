"""
src/webcam.py

Real-time facial emotion recognition from laptop webcam.

Pipeline per frame:
  1. Read frame from cv2.VideoCapture(0)
  2. Convert to grayscale
  3. Detect faces with OpenCV Haar cascade
  4. For EACH detected face:
     a. Crop face
     b. Resize to 48x48
     c. Normalize to [0, 1]
     d. Add batch+channel dims
     e. Model.predict() -> 7 probabilities
     f. Pick highest-probability emotion
     g. Apply temporal smoothing
     h. Draw bounding box + emotion label + confidence
  5. Display FPS
  6. Show frame in OpenCV window
  7. Exit when 'q' pressed

Clean exit: release camera, destroy windows.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

try:
    from src.model import EMOTION_LABELS
    from src.preprocess import load_preprocessed
except ModuleNotFoundError:
    from model import EMOTION_LABELS
    from preprocess import load_preprocessed
from tensorflow.keras.models import load_model

IMG_SIZE = 48


class EmotionSmoothie:
    """Temporal smoothing to reduce frame-to-frame jitter."""

    def __init__(self, window_size: int = 7):
        self.window_size = window_size
        self.history: deque[tuple[str, float]] = deque(maxlen=window_size)

    def update(self, emotion: str, confidence: float) -> tuple[str, float]:
        self.history.append((emotion, confidence))
        if len(self.history) < 2:
            return emotion, confidence
        scores: dict[str, float] = {}
        for em, conf in self.history:
            scores[em] = scores.get(em, 0.0) + conf
        best_emotion = max(scores, key=scores.get)
        best_score = scores[best_emotion]
        smoothed_confidence = best_score / len(self.history)
        return best_emotion, smoothed_confidence

    def reset(self):
        self.history.clear()


def find_cascade() -> str:
    candidates = [
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml",
        str(Path(__file__).parent.parent / "cascades" /
            "haarcascade_frontalface_default.xml"),
    ]
    for path in candidates:
        if Path(path).exists():
            return path
    return candidates[0]


def detect_faces(
    gray: np.ndarray,
    cascade_path: str,
) -> list[tuple[int, int, int, int]]:
    cascade = cv2.CascadeClassifier(cascade_path)
    if cascade.empty():
        raise RuntimeError(f"Failed to load cascade: {cascade_path}")
    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(40, 40),
    )
    return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]


def preprocess_face(face_bgr: np.ndarray) -> np.ndarray:
    """Crop, resize 48x48, grayscale, normalize, add dims -> (1,48,48,1)."""
    face_resized = cv2.resize(face_bgr, (IMG_SIZE, IMG_SIZE))
    if len(face_resized.shape) == 3:
        face_gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
    else:
        face_gray = face_resized
    face_norm = face_gray.astype(np.float32) / 255.0
    return np.expand_dims(face_norm, axis=(0, -1))


def draw_result(
    frame: np.ndarray,
    bbox: tuple[int, int, int, int],
    emotion: str,
    confidence: float,
) -> np.ndarray:
    x, y, w, h = bbox
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    label = f"{emotion}  {confidence*100:.1f}%"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale,
                                                  thickness)
    bg_x = x
    bg_y = y + h + 8
    bg_top = bg_y - text_h - baseline - 4
    bg_bottom = bg_y + 2
    cv2.rectangle(frame, (bg_x, bg_top), (bg_x + text_w + 6, bg_bottom),
                  (0, 255, 0), -1)
    cv2.putText(frame, label, (bg_x + 3, bg_y - 3),
                font, font_scale, (0, 0, 0), thickness + 2)
    cv2.putText(frame, label, (bg_x + 3, bg_y - 3),
                font, font_scale, (255, 255, 255), thickness)
    return frame


def run_webcam(
    model: tf.keras.Model,
    cascade_path: str,
    smoothing_window: int = 7,
    camera_index: int = 0,
    debug: bool = False,
):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("\nError: Could not open webcam.", file=sys.stderr)
        print("Check camera permissions or whether another app is using the camera.",
              file=sys.stderr)
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    ret, _ = cap.read()
    if not ret:
        print("\nError: Could not read from webcam.", file=sys.stderr)
        cap.release()
        sys.exit(1)

    face_smoothers: dict[str, EmotionSmoothie] = {}
    frame_times: deque[float] = deque(maxlen=30)

    print(f"\n[webcam] Camera opened (index {camera_index}).")
    print(f"[webcam] Resolution: {cap.get(cv2.CAP_PROP_FRAME_WIDTH):.0f}x"
          f"{cap.get(cv2.CAP_PROP_FRAME_HEIGHT):.0f}")
    print(f"[webcam] Cascade: {Path(cascade_path).name}")
    print(f"[webcam] Smoothing window: {smoothing_window} frames")
    print(f"[webcam] Debug mode: {debug}")
    print(f"[webcam] Press 'q' to quit.\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                if time.time() - (frame_times[-1] if frame_times else 0) > 5:
                    print("\n[webcam] Lost camera feed.", file=sys.stderr)
                    break
                time.sleep(0.05)
                continue

            frame_times.append(time.time())

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detect_faces(gray, cascade_path)

            for (x, y, w, h) in faces:
                face_key = f"{x}_{y}_{w}_{h}"
                if face_key not in face_smoothers:
                    face_smoothers[face_key] = EmotionSmoothie(smoothing_window)

                smoother = face_smoothers[face_key]
                face_crop = frame[y:y+h, x:x+w]
                if face_crop.size == 0:
                    continue

                face_input = preprocess_face(face_crop)
                probs = model(face_input, training=False).numpy()[0]
                pred_idx = int(np.argmax(probs))
                raw_emotion = EMOTION_LABELS[pred_idx]
                raw_confidence = float(probs[pred_idx])

                smoothed_emotion, smoothed_confidence = smoother.update(
                    raw_emotion, raw_confidence
                )

                if debug:
                    print(f"\n[DEBUG] Face: {w}x{h} | input: {model.input_shape}")
                    print("[DEBUG] Raw probs:")
                    for i, em in enumerate(EMOTION_LABELS):
                        print(f"  {em:<10}: {probs[i]*100:6.2f}%")
                    print(f"[DEBUG] Predicted: {raw_emotion} ({raw_confidence*100:.1f}%)")
                    print(f"[DEBUG] Smoothed : {smoothed_emotion} ({smoothed_confidence*100:.1f}%)",
                          flush=True)

                draw_result(frame, (x, y, w, h),
                            smoothed_emotion, smoothed_confidence)

            elapsed = (frame_times[-1] - frame_times[0]
                       if len(frame_times) > 1 else 0)
            fps = (len(frame_times) - 1) / elapsed if elapsed > 0 else 0.0

            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

            cv2.imshow("Facial Emotion Recognition  [Press 'q' to quit]", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                print("\n[webcam] Quit requested.", file=sys.stderr)
                break

    except KeyboardInterrupt:
        print("\n[webcam] Interrupted.", file=sys.stderr)
    except Exception as e:
        print(f"\n[webcam] Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[webcam] Camera released. Windows closed.", file=sys.stderr)


def cli():
    parser = argparse.ArgumentParser(
        description="Real-time facial emotion recognition from webcam"
    )
    parser.add_argument("--model", type=str,
                        default="models/emotion_model.keras",
                        help="Path to trained .keras model")
    parser.add_argument("--cascade", type=str, default=None,
                        help="Path to Haar cascade XML")
    parser.add_argument("--smoothing", type=int, default=7,
                        help="Temporal smoothing window (frames)")
    parser.add_argument("--camera", type=int, default=0,
                        help="Camera index (0=default webcam)")
    parser.add_argument("--debug", action="store_true",
                        help="Print per-face prediction details to console")
    args = parser.parse_args()

    print("=" * 60)
    print("  Facial Emotion Recognition — Real-Time Webcam")
    print("=" * 60)

    print(f"\nTensorFlow GPU available: {len(tf.config.list_physical_devices('GPU')) > 0}")

    model = load_model(args.model)
    cascade = args.cascade or find_cascade()

    run_webcam(
        model=model,
        cascade_path=cascade,
        smoothing_window=args.smoothing,
        camera_index=args.camera,
        debug=args.debug,
    )


if __name__ == "__main__":
    cli()
