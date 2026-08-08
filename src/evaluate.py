"""
Evaluate a trained model on the test set: classification report + confusion matrix.

Usage:
    python -m src.evaluate
"""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from src import config
from src.preprocess import build_datasets


def main():
    print(f"Loading model from {config.MODEL_PATH} ...")
    model = tf.keras.models.load_model(config.MODEL_PATH)

    with open(config.LABELS_PATH) as f:
        class_names = json.load(f)

    _, _, test_ds, _ = build_datasets()

    y_true = []
    y_pred = []
    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_pred.extend(np.argmax(preds, axis=1))
        y_true.extend(labels.numpy())

    print("\nClassification report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix — Waste Classification")
    plt.tight_layout()
    plt.savefig(config.CONFUSION_MATRIX_PATH)
    print(f"\nConfusion matrix saved to {config.CONFUSION_MATRIX_PATH}")


if __name__ == "__main__":
    main()
