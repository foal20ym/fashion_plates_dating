import tensorflow as tf
import numpy as np
import pandas as pd
import os
import keras_cv


def create_dataset(files):
    """Create a dataset from CSV files containing image paths and labels."""
    image_paths = []
    labels = []
    for file in files:
        df = pd.read_csv(file)
        image_paths.extend([p if p.startswith("data/") else os.path.join("data", p) for p in df["file"].tolist()])
        labels.extend(df["year"].tolist())
    return image_paths, labels


# Data augmentation definitions
data_augmentation = tf.keras.models.Sequential(
    [
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomContrast(0.1),
    ]
)

rand_augment = keras_cv.layers.RandAugment(
    value_range=(0, 255),
    augmentations_per_image=3,
    magnitude=0.8,
)


def load_and_preprocess_image(path, label, image_size, augment=False):
    """Load and preprocess an image for model input."""
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, image_size)
    if augment:
        image = tf.cast(image, tf.uint8)
        image = rand_augment(image)
        image = tf.cast(image, tf.float32) / 255.0
    else:
        image = tf.cast(image, tf.float32) / 255.0
    return image


def get_tf_dataset(
    files,
    config,
    regression=False,
    class_to_idx=None,
    min_year=None,
    max_year=None,
    augment=False,
    image_size=(224, 224),
    shuffle=True,
    include_paths=False,
):
    """Create a TensorFlow dataset from files."""
    image_paths, labels = create_dataset(files)
    image_paths = tf.constant(image_paths)

    if regression:
        if not config.get("model", {}).get("normalize_years", True):
            # No normalization, use raw years
            labels = tf.constant(labels, dtype=tf.float32)
        else:
            # Choose normalization method from config
            norm_method = config.get("model", {}).get("normalization_method", "minmax")

            if norm_method == "minmax":
                # Min-max normalization to [0, 1] range (original method)
                labels = [(y - min_year) / (max_year - min_year) for y in labels]
            elif norm_method == "standardize":
                # Standardization: zero mean and unit variance
                mean_year = np.mean(labels)
                std_year = np.std(labels)
                labels = [(y - mean_year) / std_year for y in labels]
            elif norm_method == "scale_to_range":
                # Scale to [-1, 1] range
                mean_year = (min_year + max_year) / 2
                labels = [(y - mean_year) / (max_year - min_year) * 2 for y in labels]
            else:
                # Default to min-max if method not recognized
                labels = [(y - min_year) / (max_year - min_year) for y in labels]

            # Apply label smoothing for regression during training
            if augment and config.get("training", {}).get("label_smoothing", {}).get("enabled", False):
                noise_factor = config.get("training", {}).get("label_smoothing", {}).get("factor", 0.01)
                random_noise = np.random.normal(0, noise_factor, len(labels))
                labels = [label + noise for label, noise in zip(labels, random_noise)]
                # Clip values to valid range
                labels = [max(0.0, min(1.0, label)) for label in labels]

            labels = tf.constant(labels, dtype=tf.float32)
    else:
        # Classification labels remain the same
        labels = [class_to_idx[y] for y in labels]
        labels = tf.constant(labels, dtype=tf.int32)

    ds = tf.data.Dataset.from_tensor_slices((image_paths, labels, image_paths))

    ds = ds.map(
        lambda x, y, z: (load_and_preprocess_image(x, y, image_size, augment=augment), y, z),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    if shuffle:
        ds = ds.shuffle(buffer_size=len(image_paths))

    if not include_paths:
        ds = ds.map(lambda x, y, z: (x, y))

    return ds


def prepare_datasets(train_ds, val_ds, val_ds_unshuffled, batch_size):
    """Prepare batched datasets for training and evaluation"""
    train_ds_batched = train_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_ds_batched = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_ds_unshuffled_batched = val_ds_unshuffled.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return train_ds_batched, val_ds_batched, val_ds_unshuffled_batched
