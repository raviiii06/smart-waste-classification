"""
Model architecture: transfer learning on top of a pretrained backbone.
"""

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0, MobileNetV2, ResNet50

from src import config


def build_model(num_classes, base_model_name=None):
    base_model_name = base_model_name or config.BASE_MODEL_NAME
    input_shape = config.IMG_SIZE + (3,)

    if base_model_name == "EfficientNetB0":
        base_model = EfficientNetB0(include_top=False, weights="imagenet", input_shape=input_shape)
    elif base_model_name == "MobileNetV2":
        base_model = MobileNetV2(include_top=False, weights="imagenet", input_shape=input_shape)
    elif base_model_name == "ResNet50":
        base_model = ResNet50(include_top=False, weights="imagenet", input_shape=input_shape)
    else:
        raise ValueError(f"Unknown base model: {base_model_name}")

    base_model.trainable = False  # freeze for initial head training

    inputs = tf.keras.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    return model, base_model


def unfreeze_top_layers(base_model, num_layers=30):
    """Unfreeze the last `num_layers` of the base model for fine-tuning."""
    base_model.trainable = True
    for layer in base_model.layers[:-num_layers]:
        layer.trainable = False
    return base_model
