# Facial Emotion Recognition — Real-Time Deep Learning System

## Overview

A complete college-level Deep Learning project that performs **real-time facial emotion recognition from a laptop webcam** using a CNN trained on the FER-2013 dataset.

Pipeline:
```
Webcam → OpenCV Face Detection → Crop → 48×48 Grayscale → CNN → 7 Emotions
```

## 7 Emotions

| Label | Emotion |
|-------|---------|
| 0 | Angry |
| 1 | Disgust |
| 2 | Fear |
| 3 | Happy |
| 4 | Sad |
| 5 | Surprise |
| 6 | Neutral |

## Dataset

**FER-2013** (Kaggle): 35,887 grayscale 48×48 pixel face images across 7 emotion classes.

Download `fer2013.csv` from Kaggle and place it in `dataset/`.

## Quick Start (Windows)

```bash
# 1. Create virtual environment
python -m venv venv
venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download dataset → place fer2013.csv in dataset/

# 4. Preprocess
python src/preprocess.py

# 5. Train
python src/train.py

# 6. Evaluate
python src/evaluate.py

# 7. REAL-TIME webcam demo (PRIMARY FEATURE)
python src/webcam.py

# 8. Gradio web app (live browser webcam + upload + Grad-CAM)
python app.py

# 9. Predict on a single image
python src/predict.py path/to/image.jpg
```

## Project Structure

```
facial-emotion-recognition/
├── dataset/                  # CSV + preprocessed .npy files
│   ├── fer2013.csv
│   ├── X_train.npy
│   ├── y_train.npy
│   ├── X_val.npy
│   ├── y_val.npy
│   ├── X_test.npy
│   └── y_test.npy
├── models/
│   └── emotion_model.keras   # Trained model (auto-saved during training)
├── results/
│   ├── training_history.png      # Accuracy & loss curves
│   ├── training_history.json     # History data
│   ├── confusion_matrix.png      # Confusion matrix
│   └── metrics.json              # Evaluation metrics (optional)
├── src/
│   ├── preprocess.py       # CSV parsing, normalization, train/val/test split
│   ├── model.py            # CNN architecture (and optional transfer model)
│   ├── train.py            # Training loop with EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
│   ├── evaluate.py         # Test evaluation, precision/recall/F1, confusion matrix
│   ├── predict.py          # Single-image prediction with face detection
│   ├── webcam.py           # REAL-TIME webcam emotion recognition
│   └── gradcam.py          # Grad-CAM heatmap generation
├── app.py                  # Gradio web application for Hugging Face Spaces
├── requirements.txt        # Python dependencies
├── README.md               # This file
└── .gitignore
```

## CNN Architecture

```
Input(48, 48, 1)          # Grayscale face
  ↓
Conv2D(64, 3×3)           # 64 filters learn low-level features (edges, corners)
  ↓
BatchNormalization         # Stabilizes training, allows higher learning rate
  ↓
ReLU                       # Non-linearity: max(0, x)
  ↓
MaxPooling2D(2×2)         # Halves spatial dims: 48→24, adds translation invariance
  ↓
Dropout(0.3)              # Prevents overfitting by randomly dropping neurons
  ↓
Conv2D(128, 3×3)          # 128 filters learn mid-level features (eyes, mouth shapes)
  ↓
BatchNormalization
  ↓
ReLU
  ↓
MaxPooling2D(2×2)         # 24→12
  ↓
Dropout(0.4)
  ↓
Conv2D(256, 3×3)          # 256 filters learn high-level features (facial configurations)
  ↓
BatchNormalization
  ↓
ReLU
  ↓
MaxPooling2D(2×2)         # 12→6
  ↓
Dropout(0.4)
  ↓
GlobalAveragePooling2D    # 6×6×256 → 256 (replaces Flatten, fewer params, mild regularizer)
  ↓
Dense(256)                # Classifier head
  ↓
ReLU
  ↓
Dropout(0.5)              # Heavy dropout before final classification
  ↓
Dense(7, Softmax)         # 7 emotion probabilities summing to 1.0
```

**Total parameters:** ~330K.

### Why this design?

| Component | Purpose |
|-----------|---------|
| Conv2D blocks | Extract spatial features hierarchically: edges → textures → facial parts. Filters double as spatial size halves to keep total activation volume manageable. |
| BatchNorm | Normalizes layer inputs to mean 0, variance 1. Smooths the loss landscape, allows higher LR, reduces sensitivity to initialization, acts as mild regularizer. |
| ReLU | Introduces non-linearity. Without it the network is just a linear stack of matrix multiplications. Cheaper than sigmoid/tanh, avoids vanishing gradients for positive values. |
| MaxPooling | Downsamples feature maps. Reduces computation, adds translation invariance (network focuses on "what" not "where"), forces learning of increasingly abstract features at each block. |
| Dropout | Randomly zeroes neuron outputs during training. Prevents co-adaptation of neurons, reducing overfitting. Rate increases deeper (0.3→0.4→0.5) because deeper layers are more prone to memorization. |
| GlobalAveragePooling2D | Replaces Flatten. Each 6×6 feature map → single number. 256 output values vs 9216 from Flatten. Far fewer parameters into Dense layer, and a mild regularizer (forces each feature map to be globally meaningful). |
| Dense(7, Softmax) | Converts 256-D feature vector into 7 class probabilities. Softmax ensures outputs sum to 1.0 — interpretable as confidence distribution. |

## Data Augmentation

Applied only to training data via Keras preprocessing layers (active during training, inert during inference):

| Transform | Parameters | Why |
|-----------|-----------|-----|
| Horizontal flip | Random | Faces are roughly symmetric; flipping doubles effective data without changing emotion |
| Rotation | ±18° (0.1 factor) | Small head tilts occur naturally. Larger rotations would change facial geometry unrealistically. |
| Zoom | ±10% | Simulates variation in face distance/size. Larger zoom would cut off facial features. |
| Translation | ±10% | Simulates face position variation within frame. |

These are conservative transforms — they augment diversity without creating unrealistic facial expressions.

## Training

```bash
python src/train.py
```

**Configuration (via argparse):**
- Batch size: 64
- Max epochs: 80
- Learning rate: 1e-3 (Adam)
- Random seed: 42 (configurable)

**Callbacks:**

1. **EarlyStopping** (patience=12, restore_best_weights=True)
   - Monitors validation loss. Stops when no improvement for 12 epochs.
   - Restores weights from the best epoch.
   - Prevents overfitting by not training beyond the point of diminishing returns.

2. **ReduceLROnPlateau** (factor=0.5, patience=6, min_lr=1e-6)
   - Halves learning rate when validation loss plateaus for 6 epochs.
   - High LR early → broad exploration. Lower LR later → fine-grained convergence.

3. **ModelCheckpoint** (save_best_only=True)
   - Saves model only when validation loss improves.
   - Keeps the best model, not necessarily the final epoch.

**Data pipeline:** `tf.data.Dataset` with shuffling, batching, prefetching, and augmentation applied only to training data.

**Expected output:**
```
============================================================
  Facial Emotion Recognition — Training
============================================================

[train] TensorFlow GPU available: False
[train]   No GPU detected. Training on CPU.

[train] Building model...
[train] Input shape: (48, 48, 1)
[train] Training samples: 24342
[train] Validation samples: 4299
[train] Batch size: 64
[train] Max epochs: 80
[train] Learning rate: 0.001

Model: "EmotionCNN"
... (model summary) ...
Total params: 330,xxx

[train] Starting training...
------------------------------------------------------------
Epoch 1/80
1152/1152 [==============================] - Xs/Xs
...
Epoch N/80
...
------------------------------------------------------------
[train] Training completed in Xs
[train] Best model saved to: models/emotion_model.keras
```

## Evaluation

```bash
python src/evaluate.py
```

Reports:
- Test loss and accuracy
- Per-class precision, recall, F1-score
- Full classification report
- Confusion matrix (normalized by true class, saved to `results/confusion_matrix.png`)
- Top confused emotion pairs

**Interpreting the confusion matrix:**
- Rows = true emotion, columns = predicted emotion
- Diagonal = correct predictions
- Off-diagonal = misclassifications
- High off-diagonal values = commonly confused emotions (e.g., Fear ↔ Surprise, Sad ↔ Neutral)

## Real-Time Webcam Recognition

```bash
python src/webcam.py
```

**What it does:**
1. Opens default webcam (`cv2.VideoCapture(0)`)
2. Reads frames continuously
3. Converts each frame to grayscale
4. Detects faces with OpenCV Haar cascade
5. For EACH face:
   - Crop → resize 48×48 → grayscale → normalize [0,1] → shape (1,48,48,1)
   - `model.predict()` → 7 probabilities
   - Pick highest-probability emotion
   - Apply temporal smoothing (7-frame rolling window)
   - Draw bounding box + emotion label + confidence below face
6. Display FPS counter (top-left)
7. Exit on 'q' or Escape

**Temporal smoothing:** A rolling window of recent predictions prevents label flicker from frame-to-frame noise. Window size 7 (~0.23s at 30fps) balances stability with responsiveness.

**Multiple faces:** Supported. Each face tracked independently with its own smoothing buffer.

**Exit cleanly:** Releases camera, destroys OpenCV windows, even on KeyboardInterrupt.

## Grad-CAM

Grad-CAM (Gradient-weighted Class Activation Mapping) produces a heatmap showing which regions of the face most influenced the model's prediction.

**How it works:**
1. Forward pass through the model to get final conv layer feature maps and predictions
2. Backward pass: compute gradients of the predicted class score w.r.t. feature maps
3. Weight each feature map by the global average of its gradient
4. Weighted sum of feature maps → coarse heatmap
5. ReLU the heatmap (keep only positive contributions)
6. Normalize to [0,1], resize to input size, overlay on original image

**Files:** `src/gradcam.py` (standalone), integrated into `app.py` (interactive).

**Important caveat:** Grad-CAM shows *where the model looks*, not what biologically causes the emotion. A heatmap on the mouth for "Happy" means the model used mouth features for its decision — it does NOT prove the mouth causes happiness.

## Gradio Web Application (`app.py`)

```bash
python app.py
```

Opens `http://127.0.0.1:7860` (and generates a free public `.gradio.live` link) with 4 tabs:

| Tab | Function |
|-----|----------|
| 🎥 Live Expression | Continuous browser webcam streaming with live emotion predictions & FPS |
| 📸 Upload Image | Upload an image → face detection → emotion prediction with probability distribution |
| 🔍 Grad-CAM | Upload image → Grad-CAM heatmap + overlay for face regions |
| 📊 Model Metrics & System Info | Model architecture, test accuracy, FER-2013 dataset details |

## GPU Support

```
[train] TensorFlow GPU available: False
[train]   No GPU detected. Training on CPU.
```

**Windows note:** TensorFlow ≥2.11 does not ship GPU binaries for native Windows. To use GPU:
- WSL2 + CUDA/cuDNN, or
- `pip install tensorflow-directml-plugin`

CPU training is feasible for 48×48 grayscale images (~330K params) — typically 10-30 minutes on a modern CPU. For inference (webcam, prediction), CPU is fast enough for real-time.

## Common Errors and Fixes

| Error | Cause | Fix |
|-------|-------|-----|
| `FileNotFoundError: fer2013.csv` | Dataset not downloaded | Download from Kaggle, place in `dataset/` |
| `FileNotFoundError: emotion_model.keras` | Model not trained | Run `python src/train.py` first |
| Webcam won't open | Camera in use or no camera | Close other apps. Check permissions. Try `--camera 1` for external webcam. |
| No face detected | Poor lighting, extreme angle, occlusion | Improve lighting, face forward. Haar cascade works best on frontal faces. |
| Label flickering in webcam | Small prediction fluctuations | Increase `--smoothing` (try 10-15). Default 7 is a good balance. |
| Low accuracy | Preprocessing failed, model not trained enough | Check .npy files exist. Check normalization [0,1]. Verify labels 0-6. Train longer. |
| Out of memory | Batch size too large | Reduce `--batch-size` to 32 or 16. |
| GPU not available on Windows | TF ≥2.11 Windows has no GPU binaries | Use WSL2+CUDA or DirectML plugin. Not required — CPU works fine for 48×48. |

## Deep Learning Concepts Demonstrated

This project is a practical demonstration of the following concepts:

### Foundational
1. **Tensors** — Multi-dimensional arrays. Images stored as (batch, height, width, channels) arrays. Throughout the codebase.
2. **Forward propagation** — Data flows from input through layers to output. `model.predict()` executes forward propagation. `model.fit()` does forward + backward passes.
3. **Backpropagation** — TensorFlow computes gradients automatically via `tf.GradientTape` during `model.fit()`. We never manually compute gradients — Keras handles the chain rule through the computation graph.

### Convolutional Neural Networks
4. **Convolution** — Sliding a learnable filter over the input to produce a feature map. `Conv2D` layers in `src/model.py`. Each filter detects a specific pattern.
5. **Filters / Kernels** — Small weight matrices (3×3 here) learned during training. Early layers learn edges and corners; deeper layers learn complex facial features.
6. **Feature maps** — Output of applying one filter to the input. 64 after conv1, 128 after conv2, 256 after conv3. Each map highlights where a specific feature is found.
7. **ReLU (Rectified Linear Unit)** — `max(0, x)`. Adds non-linearity. Without activation functions, a stack of linear layers collapses to a single linear transformation. ReLU is cheap and avoids vanishing gradients for positive inputs.
8. **Pooling (MaxPooling)** — Downsamples by taking the maximum in each window. Reduces spatial dimensions, adds translation invariance, reduces computation. Here: 2×2 pooling halves dimensions each block (48→24→12→6).
9. **Softmax** — Converts raw logits into probabilities that sum to 1. Used in the final Dense layer. Interpretable as confidence distribution across classes.

### Network Components
10. **Dense (Fully Connected) layers** — Every neuron connects to every neuron in the previous layer. Used in the classifier head (Dense(256) → Dense(7)).
11. **Batch Normalization** — Normalizes layer inputs to zero mean and unit variance per mini-batch. Stabilizes training, allows higher learning rates, reduces internal covariate shift, acts as mild regularizer. Applied after each Conv2D and after the first Dense layer.
12. **Dropout** — Randomly sets neurons to zero during training (with probability p). Prevents co-adaptation — neurons can't rely on specific other neurons. Rates increase deeper: 0.3 (block1) → 0.4 (block2-3) → 0.5 (classifier).

### Training Dynamics
13. **Loss functions** — `sparse_categorical_crossentropy` measures the difference between predicted probability distribution and true label. Lower loss = better predictions. "Sparse" because labels are integers (0-6), not one-hot vectors.
14. **Gradient descent** — The optimizer adjusts weights in the direction that reduces loss. Adam is used here — it adapts the learning rate per parameter.
15. **Adam optimizer** — Combines momentum (accelerates in consistent gradient directions) and RMSProp (per-parameter adaptive learning rates). Default choice for most CNN training. Used with initial LR 1e-3.
16. **Learning rate** — Step size for weight updates. Too high → unstable training (oscillating loss). Too low → slow convergence. Started at 1e-3, reduced by half when plateauing.
17. **Epochs** — One complete pass through the entire training dataset. Max 80, but EarlyStopping typically stops earlier when validation performance stops improving.
18. **Batches** — Subset of training data processed before each weight update. Batch size 64. Smaller batches → noisier gradients (better generalization) but slower convergence per epoch. Larger batches → stable gradients but may generalize worse.

### Regularization
19. **Overfitting** — Model memorizes training data but fails to generalize. Identified when training accuracy keeps rising but validation accuracy plateaus or drops. Training loss falls but validation loss rises.
20. **Underfitting** — Model too simple to capture patterns. Both training and validation accuracy stay low. Fix: add capacity (more layers/filters) or train longer.
21. **Early stopping** — Monitors validation loss. Stops training when it stops improving (patience=12 epochs). Prevents overfitting by not training beyond the optimal point. `restore_best_weights=True` rolls back to the best epoch.
22. **Learning rate scheduling** — ReduceLROnPlateau: when validation loss plateaus, halve the LR. High LR early for exploration, lower LR later for fine convergence to a better minimum.
23. **Data augmentation** — Artificially expands training data with realistic transformations (flip, rotate, zoom, translate). Reduces overfitting by exposing the model to varied versions of each image. Applied only to training data.
24. **Batch normalization as regularizer** — The noise from batch statistics during training acts as a mild regularizer, similar to dropout but softer.

### Evaluation
25. **Precision** — Of all predictions for class X, how many were correct? High precision = few false positives for that emotion.
26. **Recall** — Of all actual X samples, how many did we find? High recall = few false negatives.
27. **F1-score** — Harmonic mean of precision and recall. Balances both. Especially useful for imbalanced classes (Disgust is only ~1.5% of FER-2013).
28. **Confusion matrix** — Table showing true vs predicted classes. Reveals which emotions are confused with each other (e.g., Fear often confused with Surprise). Normalized rows show per-class recall.
29. **Grad-CAM** — Gradient-weighted Class Activation Mapping. Produces a coarse heatmap showing which input regions most influenced the prediction. Uses gradients of the class score w.r.t. final conv layer feature maps. Built in `src/gradcam.py` and integrated into `app.py`.

### Transfer Learning
30. **Transfer learning** — Reusing a model pretrained on a large dataset (ImageNet) for a new task. The pretrained backbone already learned generic visual features (edges, textures, shapes). `build_transfer_model()` in `src/model.py` uses MobileNetV2.
31. **Fine-tuning** — Optionally unfreezing some backbone layers and training with a very low learning rate to adapt pretrained features to the target domain. Implemented but frozen-by-default.
32. **When transfer learning helps vs. hurts** — On FER-2013 (48×48 grayscale, 35K images), a from-scratch CNN can match or beat transfer models because ImageNet features are tuned for 224×224 RGB natural images, not tiny grayscale faces. Transfer learning shines when the target dataset is very small or similar to the pretraining domain.

## Training Pipeline Explained

### Step 1: Preprocess (`src/preprocess.py`)

```
fer2013.csv (35,887 rows)
  ↓
Parse each row:
  - emotion: int 0-6
  - pixels: 2304 space-separated values → reshape to (48, 48)
  - Usage: 'Training' or 'PublicTest'
  ↓
Normalize: pixel / 255.0 → values in [0, 1]
  ↓
Split:
  - Training rows (28,709) → train (85% = 24,342) + val (15% = 4,299)
    Stratified by label to preserve class proportions
  - PublicTest rows (3,589) → test set
  ↓
Add channel dim: (N, 48, 48) → (N, 48, 48, 1)
  ↓
Save as .npy files in dataset/
```

**Why normalize to [0,1]?** Neural networks train faster and more stably when inputs are small, roughly zero-centered values. Pixel values 0-255 → divide by 255 → [0,1].

**Why stratified split?** FER-2013 is imbalanced (Disgust is only 1.5%). Without stratification, the validation set might have zero Disgust samples, making validation metrics meaningless.

### Step 2: Build Model (`src/model.py`)

The CNN architecture described above. Built with the Keras Functional API for clarity.

### Step 3: Train (`src/train.py`)

```
tf.data.Dataset from .npy files
  → Shuffle (buffer = dataset size, seed=42)
  → Batch (size=64)
  → Map: data_augmentation (training=True) — only on training data
  → Prefetch (AUTOTUNE — overlaps preprocessing with training)

model.fit(train_dataset, val_dataset, epochs=80, callbacks=[...])
```

### Step 4: Evaluate (`src/evaluate.py`)

Load best model from `models/emotion_model.keras`, run on test set, generate metrics and confusion matrix.

## Model Performance Expectations

FER-2013 is a challenging dataset (low resolution, in-the-wild images, class imbalance). Typical results for a from-scratch CNN:

- **Accuracy:** 65-75% on test set
- **Best classes:** Happy (easily recognized), Surprise
- **Hardest classes:** Disgust (very few training samples), Fear ↔ Surprise confusion, Sad ↔ Neutral confusion

These are baselines — the project is about the complete pipeline and understanding, not SOTA accuracy.

## Future Improvements

- Better face detection (MTCNN, RetinaFace, DNN-based)
- Transfer learning comparison (MobileNetV2, EfficientNet, ResNet) — already scaffolded in `model.py`
- Larger/better datasets (AffectNet, RAF-DB)
- Advanced augmentation (RandAugment, Cutout, GAN-generated samples for Disgust)
- Grad-CAM in real-time (optimized, threaded)
- TFLite/ONNX conversion for mobile deployment
- Face tracking (SORT, ByteTrack) to avoid re-detecting every frame
- Temporal modeling (LSTM over face sequences — expressions evolve over time)

## Technologies

| Technology | Version | Role |
|-----------|---------|------|
| Python | 3.11 | Programming language |
| TensorFlow | 2.21 | Deep learning framework (Keras integrated) |
| OpenCV | 5.0 | Face detection, image processing, webcam |
| NumPy | 2.4 | Numerical operations, array manipulation |
| Pandas | 3.0 | CSV data loading |
| Matplotlib | 3.11 | Training plots |
| Seaborn | 0.13 | Confusion matrix visualization |
| scikit-learn | 1.9 | Metrics (precision, recall, F1, confusion matrix) |
| Gradio | 6.26 | Web application |

**No PyTorch** — this project uses TensorFlow/Keras exclusively.

## License

Educational purpose. FER-2013 dataset from Kaggle — check dataset license for usage terms.
