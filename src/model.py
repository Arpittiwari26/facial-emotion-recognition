"""
src/model.py

CNN architecture for facial emotion recognition.

48x48 grayscale input, 7 emotion classes.
3 conv blocks (64→128→256) with BatchNorm, ReLU, MaxPool, Dropout.
GlobalAveragePooling → Dense(256) → Dropout → Dense(7, softmax).
"""

from __future__ import annotations

import tensorflow as tf


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


def build_emotion_model(
    input_shape: tuple = (48, 48, 1),
    num_classes: int = NUM_CLASSES,
) -> tf.keras.Model:
    """Build high-accuracy FER CNN architecture preserving spatial facial layout."""
    inputs = tf.keras.Input(shape=input_shape, name="input_image")

    # Block 1: 48x48x1 → 24x24x32
    x = tf.keras.layers.Conv2D(32, (3, 3), padding="same", kernel_initializer="he_normal", name="conv1_1")(inputs)
    x = tf.keras.layers.BatchNormalization(name="bn1_1")(x)
    x = tf.keras.layers.Activation("relu", name="relu1_1")(x)
    x = tf.keras.layers.Conv2D(32, (3, 3), padding="same", kernel_initializer="he_normal", name="conv1_2")(x)
    x = tf.keras.layers.BatchNormalization(name="bn1_2")(x)
    x = tf.keras.layers.Activation("relu", name="relu1_2")(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name="pool1")(x)
    x = tf.keras.layers.Dropout(0.2, name="drop1")(x)

    # Block 2: 24x24x32 → 12x12x64
    x = tf.keras.layers.Conv2D(64, (3, 3), padding="same", kernel_initializer="he_normal", name="conv2_1")(x)
    x = tf.keras.layers.BatchNormalization(name="bn2_1")(x)
    x = tf.keras.layers.Activation("relu", name="relu2_1")(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), padding="same", kernel_initializer="he_normal", name="conv2_2")(x)
    x = tf.keras.layers.BatchNormalization(name="bn2_2")(x)
    x = tf.keras.layers.Activation("relu", name="relu2_2")(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name="pool2")(x)
    x = tf.keras.layers.Dropout(0.25, name="drop2")(x)

    # Block 3: 12x12x64 → 6x6x128
    x = tf.keras.layers.Conv2D(128, (3, 3), padding="same", kernel_initializer="he_normal", name="conv3_1")(x)
    x = tf.keras.layers.BatchNormalization(name="bn3_1")(x)
    x = tf.keras.layers.Activation("relu", name="relu3_1")(x)
    x = tf.keras.layers.Conv2D(128, (3, 3), padding="same", kernel_initializer="he_normal", name="conv3_2")(x)
    x = tf.keras.layers.BatchNormalization(name="bn3_2")(x)
    x = tf.keras.layers.Activation("relu", name="relu3_2")(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name="pool3")(x)
    x = tf.keras.layers.Dropout(0.3, name="drop3")(x)

    # Flatten: preserve spatial facial layout (6x6x128 = 4608 features)
    x = tf.keras.layers.Flatten(name="flatten")(x)
    x = tf.keras.layers.Dense(256, kernel_initializer="he_normal", name="dense1")(x)
    x = tf.keras.layers.BatchNormalization(name="bn4")(x)
    x = tf.keras.layers.Activation("relu", name="relu4")(x)
    x = tf.keras.layers.Dropout(0.4, name="drop4")(x)

    # Final: 7 emotions, softmax output
    outputs = tf.keras.layers.Dense(units=num_classes, activation="softmax", name="predictions")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="EmotionCNN")
    total_params = model.count_params()
    print(f"[model] Total parameters: {total_params:,}")
    return model


def compile_model(
    model: tf.keras.Model,
    learning_rate: float = 1e-3,
) -> tf.keras.Model:
    """Compile model with Adam optimizer + sparse categorical crossentropy."""
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_transfer_model(
    base_model_name: str = "MobileNetV2",
    input_shape: tuple = (48, 48, 1),
    num_classes: int = NUM_CLASSES,
) -> tf.keras.Model:
    """Optional transfer-learning model using a pre-trained backbone."""
    inputs = tf.keras.Input(shape=input_shape, name="input_image")

    x = tf.keras.layers.Resizing(160, 160, name="resize")(inputs)
    x = tf.keras.layers.Lambda(lambda t: tf.tile(t, [1, 1, 1, 3]), name="to_rgb")(x)

    base_model_map = {
        "MobileNetV2": tf.keras.applications.MobileNetV2,
        "EfficientNetB0": tf.keras.applications.EfficientNetB0,
        "ResNet50": tf.keras.applications.ResNet50,
    }
    if base_model_name not in base_model_map:
        raise ValueError(f"Unknown base model: {base_model_name}")

    Base = base_model_map[base_model_name]
    base = Base(input_shape=(160, 160, 3), include_top=False,
                weights="imagenet", pooling="avg")
    base.trainable = False

    x = base(x, training=False)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs,
                           name=f"TransferEmotion_{base_model_name}")
    return model
