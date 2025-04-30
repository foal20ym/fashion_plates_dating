from keras.applications import NASNetLarge
from keras.optimizers import Adam
from keras.layers import Flatten, Dense, Dropout
from keras.models import Model
import random
import pandas as pd
import numpy as np
import tensorflow as tf
import sys
import time

image_size = (299, 299)
# batch_size = 32


def create_dataset(files):
    image_paths = []
    labels = []
    for file in files:
        df = pd.read_csv(file)
        image_paths.extend(df["file"].tolist())
        labels.extend(df["year"].tolist())
    return image_paths, labels


def load_and_preprocess_image(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, image_size)
    image = image / 255.0
    return image, label


def get_tf_dataset(files, regression=False, class_to_idx=None):
    image_paths, labels = create_dataset(files)
    image_paths = tf.constant(image_paths)
    if regression:
        labels = tf.constant(labels, dtype=tf.float32)
    else:
        # Map years to class indices for classification
        labels = [class_to_idx[y] for y in labels]
        labels = tf.constant(labels, dtype=tf.int32)
    ds = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    ds = ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.shuffle(buffer_size=len(image_paths))
    # ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def build_model(num_classes=None, regression=False):
    base_model = NASNetLarge(include_top=False, weights="imagenet", input_shape=(299, 299, 3), pooling="avg")
    x = base_model.output
    x = Flatten(name="FLATTEN")(x)
    x = Dense(256, activation="relu", name="last_FC1")(x)
    x = Dropout(0.5, name="DROPOUT")(x)
    if regression:
        predictions = Dense(1, activation="linear", name="PREDICTIONS")(x)
    else:
        predictions = Dense(num_classes, activation="softmax", name="PREDICTIONS")(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    for layer in base_model.layers:
        layer.trainable = False
    return model


def main(task="classification"):
    start_time = time.time()

    # --- GPU Configuration for Optimal Utilization ---
    print("TensorFlow Version:", tf.__version__)
    print("Num GPUs Available:", len(tf.config.list_physical_devices("GPU")))
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            # Set memory growth to True to avoid allocating all memory at once
            # tf.config.experimental.set_memory_growth(gpus[0], True)

            # Set the visible devices to use only the first GPU (if multiple are available)
            tf.config.set_visible_devices(gpus[0], "GPU")
            print("Using GPU:", gpus[0])

            # Enable mixed precision training for potential speedup and reduced memory usage
            # policy = tf.keras.mixed_precision.Policy("mixed_float16")
            # tf.keras.mixed_precision.set_global_policy(policy)
            # print("Using mixed precision training.")

        except RuntimeError as e:
            print(e)
    else:
        print("No GPU detected. Training will run on CPU.")

    # Fold selection
    fold_nums = [num for num in range(10)]
    print(fold_nums)
    test_fold = random.choice(fold_nums)
    print(f"Test fold: {test_fold}")
    train_folds = [fold for fold in fold_nums if fold != test_fold]
    print(f"Train folds: {train_folds}")
    train_files = [f"data/datasets/fold{fold}.csv" for fold in train_folds]
    test_file = f"data/datasets/fold{test_fold}.csv"
    fold_files = train_files
    test_files = [test_file]

    regression = task == "regression"

    # For classification, build class mapping
    if not regression:
        # Collect all unique years in training set
        all_years = []
        for file in fold_files:
            df = pd.read_csv(file)
            all_years.extend(df["year"].tolist())
        classes = sorted(list(set(all_years)))
        class_to_idx = {y: i for i, y in enumerate(classes)}
        num_classes = len(classes)
    else:
        class_to_idx = None
        num_classes = None

    train_ds = get_tf_dataset(fold_files, regression=regression, class_to_idx=class_to_idx)
    val_ds = get_tf_dataset(test_files, regression=regression, class_to_idx=class_to_idx)

    print(train_ds)
    print(val_ds)

    model = build_model(num_classes=num_classes, regression=regression)

    if regression:
        model.compile(optimizer=Adam(learning_rate=0.0001), loss="mean_squared_error", metrics=["mae"])
    else:
        model.compile(
            optimizer=Adam(learning_rate=0.0001), loss="sparse_categorical_crossentropy", metrics=["accuracy"]
        )

    model.summary()

    end_time = time.time()
    running_time = end_time - start_time
    hours = int(running_time // 3600)
    minutes = int((running_time % 3600) // 60)
    seconds = int(running_time % 60)
    print(f"Running time: {hours} hours, {minutes} minutes, {seconds} seconds")


if __name__ == "__main__":
    # Usage: python fashion.py [classification|regression]
    task = sys.argv[1] if len(sys.argv) > 1 else "classification"
    main(task)
