"""
Data loading + preprocessing pipeline.

Expects a directory structure like:

    data/garbage-classification/
        cardboard/
            img001.jpg
            img002.jpg
        glass/
            ...
        metal/
            ...
        ...

This is exactly the structure of the Kaggle "Garbage Classification" dataset
(mostafaabla/garbage-classification) once unzipped.
"""

import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input as eff_preprocess
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mob_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as res_preprocess

from src import config


def get_preprocess_fn():
    """Return the correct preprocess_input function for the chosen base model."""
    if config.BASE_MODEL_NAME == "EfficientNetB0":
        return eff_preprocess
    elif config.BASE_MODEL_NAME == "MobileNetV2":
        return mob_preprocess
    elif config.BASE_MODEL_NAME == "ResNet50":
        return res_preprocess
    raise ValueError(f"Unknown base model: {config.BASE_MODEL_NAME}")


def build_datasets(data_dir=None):
    """
    Build train/val/test tf.data.Dataset objects from a directory of
    class-labeled image subfolders. Returns (train_ds, val_ds, test_ds, class_names).
    """
    data_dir = data_dir or config.DATA_DIR

    # First split: 70% train, 30% temp (val+test)
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=config.VALIDATION_SPLIT + config.TEST_SPLIT,
        subset="training",
        seed=config.SEED,
        image_size=config.IMG_SIZE,
        batch_size=config.BATCH_SIZE,
    )

    temp_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=config.VALIDATION_SPLIT + config.TEST_SPLIT,
        subset="validation",
        seed=config.SEED,
        image_size=config.IMG_SIZE,
        batch_size=config.BATCH_SIZE,
    )

    class_names = train_ds.class_names

    # Split temp_ds roughly in half -> val / test
    temp_batches = tf.data.experimental.cardinality(temp_ds).numpy()
    val_batches = temp_batches // 2
    val_ds = temp_ds.take(val_batches)
    test_ds = temp_ds.skip(val_batches)

    preprocess_fn = get_preprocess_fn()

    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.15),
        tf.keras.layers.RandomZoom(0.15),
        tf.keras.layers.RandomContrast(0.1),
    ])

    def prep_train(x, y):
        x = data_augmentation(x)
        x = preprocess_fn(x)
        return x, y

    def prep_eval(x, y):
        x = preprocess_fn(x)
        return x, y

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.map(prep_train, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
    val_ds = val_ds.map(prep_eval, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
    test_ds = test_ds.map(prep_eval, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)

    return train_ds, val_ds, test_ds, class_names
