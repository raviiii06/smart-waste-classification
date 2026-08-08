"""
Train the waste classifier.

Usage:
    python -m src.train

This runs a two-phase transfer-learning process:
  Phase 1: train only the new classification head (backbone frozen)
  Phase 2: unfreeze the top layers of the backbone and fine-tune at a low LR

Saves:
  models/waste_classifier.h5   - trained Keras model
  models/labels.json           - class index -> class name mapping
  models/training_history.png  - accuracy/loss curves
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf

from src import config
from app.model import build_model, unfreeze_top_layers
from src.preprocess import build_datasets


def plot_history(history_head, history_finetune, out_path):
    acc = history_head.history["accuracy"] + history_finetune.history["accuracy"]
    val_acc = history_head.history["val_accuracy"] + history_finetune.history["val_accuracy"]
    loss = history_head.history["loss"] + history_finetune.history["loss"]
    val_loss = history_head.history["val_loss"] + history_finetune.history["val_loss"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(acc, label="train_acc")
    axes[0].plot(val_acc, label="val_acc")
    axes[0].axvline(len(history_head.history["accuracy"]) - 0.5, color="gray", linestyle="--", label="fine-tune start")
    axes[0].set_title("Accuracy")
    axes[0].legend()

    axes[1].plot(loss, label="train_loss")
    axes[1].plot(val_loss, label="val_loss")
    axes[1].axvline(len(history_head.history["loss"]) - 0.5, color="gray", linestyle="--")
    axes[1].set_title("Loss")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_path)


def main():
    os.makedirs(config.MODEL_DIR, exist_ok=True)

    print(f"Loading dataset from {config.DATA_DIR} ...")
    train_ds, val_ds, test_ds, class_names = build_datasets()
    num_classes = len(class_names)
    print(f"Found {num_classes} classes: {class_names}")

    with open(config.LABELS_PATH, "w") as f:
        json.dump(class_names, f, indent=2)

    model, base_model = build_model(num_classes)

    # ---- Phase 1: train the head ----
    model.compile(
        optimizer=tf.keras.optimizers.Adam(config.LEARNING_RATE_HEAD),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(config.MODEL_PATH, monitor="val_accuracy", save_best_only=True),
    ]

    print("\n--- Phase 1: training classification head ---")
    history_head = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.EPOCHS_HEAD,
        callbacks=callbacks,
    )

    # ---- Phase 2: fine-tune top layers of the backbone ----
    print("\n--- Phase 2: fine-tuning backbone ---")
    unfreeze_top_layers(base_model, num_layers=30)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(config.LEARNING_RATE_FINE_TUNE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    history_finetune = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.EPOCHS_FINE_TUNE,
        callbacks=callbacks,
    )

    model.save(config.MODEL_PATH)
    print(f"\nModel saved to {config.MODEL_PATH}")

    plot_history(history_head, history_finetune, config.HISTORY_PLOT_PATH)
    print(f"Training curves saved to {config.HISTORY_PLOT_PATH}")

    # Quick test-set check
    test_loss, test_acc = model.evaluate(test_ds)
    print(f"\nTest accuracy: {test_acc:.4f} | Test loss: {test_loss:.4f}")


if __name__ == "__main__":
    main()
