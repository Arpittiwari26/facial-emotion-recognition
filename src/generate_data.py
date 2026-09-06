"""
src/generate_data.py

Generate synthetic FER-2013-like data with REALISTIC face variation:
- Variable skin tones (light to dark)
- Variable contrast (low to high)
- Variable lighting (bright to dark, left/right bias)
- Variable feature shapes (eyes, brows, mouth shapes within each emotion)
- Random small position jitter (±3px)
- Gaussian noise at multiple levels

Key insight: real FER-2013 faces have WIDE variation in appearance.
The model must learn emotion features that survive this variation.
If all synthetic faces look identical within a class, the model
memorizes exact pixel patterns and fails on real faces.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

EMOTION_LABELS = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Sad",
    "Surprise",
    "Neutral",
]

NUM_CLASSES = len(EMOTION_LABELS)
IMG_SIZE = 48


def _paint(img, x0, y0, x1, y1, color, noise=0.0):
    x0, x1 = max(0, int(x0)), min(IMG_SIZE, int(x1))
    y0, y1 = max(0, int(y0)), min(IMG_SIZE, int(y1))
    if x0 < x1 and y0 < y1:
        img[y0:y1, x0:x1] = np.full((y1-y0, x1-x0), color, dtype=np.float64)
        if noise > 0:
            img[y0:y1, x0:x1] += np.random.normal(0, noise, (y1-y0, x1-x0))


def _line(img, x0, y0, x1, y1, color, width=1, noise=0.0):
    dx, dy = abs(x1-x0), abs(y1-y0)
    steps = max(int(dx), int(dy), 1)
    for i in range(steps + 1):
        t = i / steps
        px, py = int(x0 + t*(x1-x0)), int(y0 + t*(y1-y0))
        for w in range(-int(width), int(width)+1):
            for h in range(-int(width), int(width)+1):
                if w*w + h*h <= int(width)*int(width):
                    sx, sy = int(px+w), int(py+h)
                    if 0 <= sx < IMG_SIZE and 0 <= sy < IMG_SIZE:
                        img[sy, sx] = color + np.random.normal(0, noise)


def _ellipse(img, cx, cy, rx, ry, color, noise=0.0):
    """Draw a filled ellipse (axis-aligned)."""
    irx, iry = int(rx), int(ry)
    for dy in range(-iry, iry+1):
        for dx in range(-irx, irx+1):
            if (dx/rx)**2 + (dy/ry)**2 <= 1.0:
                sx, sy = int(cx+dx), int(cy+dy)
                if 0 <= sx < IMG_SIZE and 0 <= sy < IMG_SIZE:
                    img[sy, sx] = color + np.random.normal(0, noise)


def _arc_face(img, cx, cy, rx, ry, start_a, end_a, color, width=1, noise=0.0):
    """Draw an arc (portion of ellipse)."""
    for angle_deg in np.linspace(start_a, end_a, 30):
        rad = np.radians(angle_deg)
        px = cx + rx * np.cos(rad)
        py = cy + ry * np.sin(rad)
        for w in range(-width, width+1):
            for h in range(-width, width+1):
                sx, sy = int(px+w), int(py+h)
                if 0 <= sx < IMG_SIZE and 0 <= sy < IMG_SIZE:
                    img[sy, sx] = color + np.random.normal(0, noise)


def generate_face(emotion: int, rng: np.random.RandomState) -> np.ndarray:
    """Generate ONE face with emotion-specific features AND realistic variation.

    Each call produces a DIFFERENT face (skin tone, contrast, lighting, feature
    shapes vary) but with emotion-consistent patterns. This teaches the model
    to recognize emotions across appearance variation — like real FER-2013.
    """
    # ── Random face parameters (variation per face) ──
    skin_base = rng.choice([90, 110, 130, 150, 170])  # skin brightness: dark→light
    skin_contrast = rng.uniform(0.5, 1.5)              # how dark features are vs skin
    lighting = rng.uniform(0.8, 1.2)                   # overall brightness multiplier
    light_side = rng.choice([-1, 0, 1])                # -1=left dark, 0=flat, 1=right dark
    noise_level = rng.uniform(3, 12)                   # background noise
    feature_jitter = rng.uniform(-3, 3, 2)             # small position shift

    jx, jy = feature_jitter

    # ── Background ──
    img = np.full((IMG_SIZE, IMG_SIZE), skin_base * lighting, dtype=np.float64)
    # Left/right lighting gradient
    if light_side != 0:
        for x in range(IMG_SIZE):
            factor = 1.0 + light_side * 0.15 * (x / IMG_SIZE - 0.5)
            img[:, x] *= factor
    img += rng.normal(0, noise_level, (IMG_SIZE, IMG_SIZE))

    # ── Face oval (skin-colored region) ──
    face_cx, face_cy = 24 + jx, 24 + jy
    _ellipse(img, face_cx, face_cy + 2, 20, 24, skin_base * lighting * 1.05, noise_level)
    # Slightly lighter forehead
    _ellipse(img, face_cx, face_cy - 8, 16, 8, skin_base * lighting * 1.1, noise_level)

    # ── Emotion-specific features ──
    if emotion == 0:  # ANGRY — angled-in brows (V), narrowed eyes, tight mouth
        # Angry brows: angled DOWN and IN (V-shape pointing down)
        brow_color = skin_base * lighting * 0.55 * skin_contrast
        _line(img, 14+jx, 12+jy, 20+jx, 15+jy, brow_color, 1, noise_level*0.5)
        _line(img, 34+jx, 12+jy, 28+jx, 15+jy, brow_color, 1, noise_level*0.5)
        # Dark eyes (narrow, intense)
        eye_color = skin_base * lighting * 0.40 * skin_contrast
        _ellipse(img, 17+jx, 18+jy, 3, 2.5, eye_color, noise_level*0.3)
        _ellipse(img, 31+jx, 18+jy, 3, 2.5, eye_color, noise_level*0.3)
        # Pupils
        _ellipse(img, 17+jx, 18+jy, 1.5, 1.5, skin_base * lighting * 0.15, noise_level*0.2)
        _ellipse(img, 31+jx, 18+jy, 1.5, 1.5, skin_base * lighting * 0.15, noise_level*0.2)
        # Nose (medium)
        nose_color = skin_base * lighting * 0.80 * skin_contrast
        _ellipse(img, 24+jx, 24+jy, 2.5, 4, nose_color, noise_level*0.4)
        # Tight, flat/straight mouth (slight frown)
        mouth_color = skin_base * lighting * 0.50 * skin_contrast
        _line(img, 18+jx, 34+jy, 30+jx, 34+jy, mouth_color, 1, noise_level*0.4)
        _paint(img, 19+jx, 34+jy, 29+jx, 35+jy, skin_base * lighting * 0.40 * skin_contrast, noise_level*0.3)

    elif emotion == 1:  # DISGUST — wrinkled nose, narrow eyes, raised mouth
        # Normal-ish brows (slightly lowered/flat)
        brow_color = skin_base * lighting * 0.65 * skin_contrast
        _line(img, 14+jx, 12+jy, 20+jx, 12+jy, brow_color, 1, noise_level*0.5)
        _line(img, 28+jx, 12+jy, 34+jx, 12+jy, brow_color, 1, noise_level*0.5)
        # Narrow eyes
        eye_color = skin_base * lighting * 0.45 * skin_contrast
        _ellipse(img, 17+jx, 18+jy, 2.5, 2, eye_color, noise_level*0.3)
        _ellipse(img, 31+jx, 18+jy, 2.5, 2, eye_color, noise_level*0.3)
        _ellipse(img, 17+jx, 18+jy, 1, 1, skin_base * lighting * 0.15, noise_level*0.2)
        _ellipse(img, 31+jx, 18+jy, 1, 1, skin_base * lighting * 0.15, noise_level*0.2)
        # WRINKLED nose — horizontal lines (distinctive disgust feature)
        nose_color = skin_base * lighting * 0.50 * skin_contrast
        for row_y in range(21+jy, 27+jy):
            _paint(img, 20+jx, row_y, 28+jx, row_y+1,
                   skin_base * lighting * (1.2 if row_y % 2 == 0 else 0.7) * skin_contrast,
                   noise_level*0.3)
        # Raised upper lip (sneer)
        lip_color = skin_base * lighting * 0.35 * skin_contrast
        _paint(img, 19+jx, 31+jy, 29+jx, 32+jy, lip_color, noise_level*0.3)
        _paint(img, 18+jx, 32+jy, 30+jx, 33+jy, skin_base * lighting * 0.55 * skin_contrast, noise_level*0.3)

    elif emotion == 2:  # FEAR — raised brows, wide eyes, open mouth
        # RAISED brows (high on forehead)
        brow_color = skin_base * lighting * 0.60 * skin_contrast
        _line(img, 14+jx, 8+jy, 20+jx, 8+jy, brow_color, 1, noise_level*0.5)
        _line(img, 28+jx, 8+jy, 34+jx, 8+jy, brow_color, 1, noise_level*0.5)
        # WIDE eyes (large, tense)
        eye_color = skin_base * lighting * 0.35 * skin_contrast
        _ellipse(img, 17+jx, 16+jy, 3.5, 3.5, eye_color, noise_level*0.3)
        _ellipse(img, 31+jx, 16+jy, 3.5, 3.5, eye_color, noise_level*0.3)
        # Iris
        _ellipse(img, 17+jx, 16+jy, 1.5, 2.5, skin_base * lighting * 0.12, noise_level*0.2)
        _ellipse(img, 31+jx, 16+jy, 1.5, 2.5, skin_base * lighting * 0.12, noise_level*0.2)
        # Nose
        nose_color = skin_base * lighting * 0.75 * skin_contrast
        _ellipse(img, 24+jx, 24+jy, 2.5, 4, nose_color, noise_level*0.4)
        # Open mouth (wide, tense)
        mouth_color = skin_base * lighting * 0.30 * skin_contrast
        _ellipse(img, 24+jx, 35+jy, 6, 3.5, mouth_color, noise_level*0.3)
        _paint(img, 20+jx, 34+jy, 28+jx, 36+jy,
               skin_base * lighting * 0.15 * skin_contrast, noise_level*0.2)

    elif emotion == 3:  # HAPPY — smile, raised cheeks, slight brow lift
        # Slightly raised flat brows
        brow_color = skin_base * lighting * 0.65 * skin_contrast
        _line(img, 14+jx, 11+jy, 20+jx, 11+jy, brow_color, 1, noise_level*0.5)
        _line(img, 28+jx, 11+jy, 34+jx, 11+jy, brow_color, 1, noise_level*0.5)
        # Normal eyes
        eye_color = skin_base * lighting * 0.45 * skin_contrast
        _ellipse(img, 17+jx, 18+jy, 2.5, 2.5, eye_color, noise_level*0.3)
        _ellipse(img, 31+jx, 18+jy, 2.5, 2.5, eye_color, noise_level*0.3)
        _ellipse(img, 17+jx, 18+jy, 1, 1.5, skin_base * lighting * 0.15, noise_level*0.2)
        _ellipse(img, 31+jx, 18+jy, 1, 1.5, skin_base * lighting * 0.15, noise_level*0.2)
        # RAISED CHEEKS (smile highlight) — HAPPY signature
        cheek_color = skin_base * lighting * 1.15
        _ellipse(img, 14+jx, 25+jy, 5, 4, cheek_color, noise_level*0.3)
        _ellipse(img, 34+jx, 25+jy, 5, 4, cheek_color, noise_level*0.3)
        # Nose
        nose_color = skin_base * lighting * 0.80 * skin_contrast
        _ellipse(img, 24+jx, 24+jy, 2.5, 4, nose_color, noise_level*0.4)
        # SMILE (curved up) — HAPPY signature
        mouth_color = skin_base * lighting * 0.30 * skin_contrast
        _arc_face(img, 24+jx, 34+jy, 7, 3.5, 200, 340, mouth_color, 1.5, noise_level*0.3)
        _paint(img, 19+jx, 33+jy, 29+jx, 35+jy,
               skin_base * lighting * 0.50 * skin_contrast, noise_level*0.3)

    elif emotion == 4:  # SAD — raised inner brows, downturned mouth, tear areas
        # SAD brows: inner corners RAISED (arch), outer corners normal/down
        brow_color = skin_base * lighting * 0.60 * skin_contrast
        _line(img, 14+jx, 14+jy, 18+jx, 10+jy, brow_color, 1, noise_level*0.5)
        _line(img, 18+jx, 10+jy, 20+jx, 12+jy, brow_color, 1, noise_level*0.5)
        _line(img, 34+jx, 14+jy, 30+jx, 10+jy, brow_color, 1, noise_level*0.5)
        _line(img, 30+jx, 10+jy, 28+jx, 12+jy, brow_color, 1, noise_level*0.5)
        # Slightly narrowed eyes (grief)
        eye_color = skin_base * lighting * 0.45 * skin_contrast
        _ellipse(img, 17+jx, 18+jy, 2.5, 2.5, eye_color, noise_level*0.3)
        _ellipse(img, 31+jx, 18+jy, 2.5, 2.5, eye_color, noise_level*0.3)
        _ellipse(img, 17+jx, 18+jy, 1, 1.5, skin_base * lighting * 0.15, noise_level*0.2)
        _ellipse(img, 31+jx, 18+jy, 1, 1.5, skin_base * lighting * 0.15, noise_level*0.2)
        # TEAR AREAS (bright patches under eyes) — SAD signature
        tear_color = skin_base * lighting * 1.2
        _ellipse(img, 15+jx, 22+jy, 3, 3, tear_color, noise_level*0.3)
        _ellipse(img, 33+jx, 22+jy, 3, 3, tear_color, noise_level*0.3)
        # Nose
        nose_color = skin_base * lighting * 0.75 * skin_contrast
        _ellipse(img, 24+jx, 24+jy, 2.5, 4, nose_color, noise_level*0.4)
        # DOWNTURNED mouth (sad)
        mouth_color = skin_base * lighting * 0.50 * skin_contrast
        _line(img, 19+jx, 34+jy, 24+jx, 34+jy, mouth_color, 1, noise_level*0.3)
        _line(img, 24+jx, 34+jy, 29+jx, 35+jy, mouth_color, 1, noise_level*0.3)
        # Downturned corners (darker)
        _paint(img, 18+jx, 34+jy, 20+jx, 35+jy, skin_base * lighting * 0.40 * skin_contrast, noise_level*0.3)
        _paint(img, 28+jx, 34+jy, 30+jx, 35+jy, skin_base * lighting * 0.40 * skin_contrast, noise_level*0.3)

    elif emotion == 5:  # SURPRISE — very raised brows, huge eyes, open mouth
        # VERY raised brows (high arch)
        brow_color = skin_base * lighting * 0.55 * skin_contrast
        _line(img, 14+jx, 7+jy, 20+jx, 7+jy, brow_color, 1, noise_level*0.5)
        _line(img, 28+jx, 7+jy, 34+jx, 7+jy, brow_color, 1, noise_level*0.5)
        # HUGE wide eyes
        eye_color = skin_base * lighting * 0.25 * skin_contrast
        _ellipse(img, 17+jx, 15+jy, 4, 4, eye_color, noise_level*0.3)
        _ellipse(img, 31+jx, 15+jy, 4, 4, eye_color, noise_level*0.3)
        # Iris (large)
        _ellipse(img, 17+jx, 15+jy, 2, 3, skin_base * lighting * 0.10, noise_level*0.2)
        _ellipse(img, 31+jx, 15+jy, 2, 3, skin_base * lighting * 0.10, noise_level*0.2)
        # Nose
        nose_color = skin_base * lighting * 0.75 * skin_contrast
        _ellipse(img, 24+jx, 24+jy, 2.5, 4, nose_color, noise_level*0.4)
        # BIG open mouth
        mouth_color = skin_base * lighting * 0.20 * skin_contrast
        _ellipse(img, 24+jx, 36+jy, 7, 4, mouth_color, noise_level*0.3)
        _paint(img, 20+jx, 34+jy, 28+jx, 38+jy,
               skin_base * lighting * 0.10 * skin_contrast, noise_level*0.2)

    else:  # NEUTRAL — flat brows, normal eyes, straight mouth
        # Flat brows
        brow_color = skin_base * lighting * 0.65 * skin_contrast
        _line(img, 14+jx, 12+jy, 20+jx, 12+jy, brow_color, 1, noise_level*0.5)
        _line(img, 28+jx, 12+jy, 34+jx, 12+jy, brow_color, 1, noise_level*0.5)
        # Normal eyes
        eye_color = skin_base * lighting * 0.45 * skin_contrast
        _ellipse(img, 17+jx, 18+jy, 2.5, 2.5, eye_color, noise_level*0.3)
        _ellipse(img, 31+jx, 18+jy, 2.5, 2.5, eye_color, noise_level*0.3)
        _ellipse(img, 17+jx, 18+jy, 1, 1.5, skin_base * lighting * 0.15, noise_level*0.2)
        _ellipse(img, 31+jx, 18+jy, 1, 1.5, skin_base * lighting * 0.15, noise_level*0.2)
        # Nose
        nose_color = skin_base * lighting * 0.80 * skin_contrast
        _ellipse(img, 24+jx, 24+jy, 2.5, 4, nose_color, noise_level*0.4)
        # Straight neutral mouth
        mouth_color = skin_base * lighting * 0.50 * skin_contrast
        _line(img, 19+jx, 34+jy, 29+jx, 34+jy, mouth_color, 1, noise_level*0.3)
        _paint(img, 18+jx, 34+jy, 30+jx, 35+jy,
               skin_base * lighting * 0.45 * skin_contrast, noise_level*0.3)

    # Final noise
    img += rng.normal(0, noise_level*0.5, (IMG_SIZE, IMG_SIZE))
    return np.clip(img, 0, 255).astype(np.float64)


def generate_synthetic_fer(num_per_class: int = 343, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic FER-2013 with realistic face variation.

    Each face has random: skin tone, contrast, lighting, feature shapes,
    and small position jitter. Emotion features are consistent but appear
    across a range of appearances — like real FER-2013.
    """
    rows = []
    master_rng = np.random.RandomState(seed)
    for emotion in range(NUM_CLASSES):
        for i in range(num_per_class):
            rng = np.random.RandomState(master_rng.randint(0, 2**31))
            face = generate_face(emotion, rng)
            pixels = " ".join(str(int(p)) for p in face.flatten())
            usage = "Training" if i < num_per_class * 0.85 else "PublicTest"
            rows.append({"emotion": emotion, "pixels": pixels, "Usage": usage})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("Generating realistic synthetic FER-2013 with appearance variation...")
    df = generate_synthetic_fer(num_per_class=343, seed=42)
    csv_path = Path("dataset/fer2013.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"Generated {len(df)} images ({len(df)//NUM_CLASSES} per class)")
    print(f"Saved to {csv_path}")
    # Show variation stats
    print(f"\nPer-class appearance stats (5 random samples each):")
    for em in range(NUM_CLASSES):
        sub = df[df['emotion'] == em]
        samples = sub.sample(5, random_state=99)
        means, stds = [], []
        for _, row in samples.iterrows():
            px = np.array([int(x) for x in row['pixels'].split()], dtype=np.float64).reshape(48,48)
            means.append(px.mean())
            stds.append(px.std())
        print(f"  {EMOTION_LABELS[em]:10s}: mean_range=[{min(means):.0f}-{max(means):.0f}]  "
              f"std_range=[{min(stds):.1f}-{max(stds):.1f}]")
    print(f"\nUsage: {df['Usage'].value_counts().to_dict()}")
