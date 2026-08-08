"""
Central configuration for the Smart Waste Classification project.
Edit these values to match your dataset location and training preferences.
"""

import os

# ---- Paths ----
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "garbage-classification","garbage_classification")  # folder with one subfolder per class
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "waste_classifier.h5")
LABELS_PATH = os.path.join(MODEL_DIR, "labels.json")
HISTORY_PLOT_PATH = os.path.join(MODEL_DIR, "training_history.png")
CONFUSION_MATRIX_PATH = os.path.join(MODEL_DIR, "confusion_matrix.png")

# ---- Class names (Kaggle "Garbage Classification" 12-class dataset) ----
# If your dataset has different classes, this list is auto-derived from folder
# names at training time — this is just a fallback/reference.
DEFAULT_CLASSES = [
    "battery", "biological", "brown-glass", "cardboard", "clothes",
    "green-glass", "metal", "paper", "plastic", "shoes", "trash", "white-glass",
]

# ---- Image / training hyperparameters ----
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS_HEAD = 10        # training only the new classification head
EPOCHS_FINE_TUNE = 10   # fine-tuning the top layers of the base model
LEARNING_RATE_HEAD = 1e-3
LEARNING_RATE_FINE_TUNE = 1e-5
VALIDATION_SPLIT = 0.15
TEST_SPLIT = 0.15
SEED = 42

# ---- Model ----
BASE_MODEL_NAME = "EfficientNetB0"  # options: EfficientNetB0, MobileNetV2, ResNet50
