from tensorflow.keras.applications import InceptionV3, ResNet101, NASNetMobile
from tensorflow.keras.optimizers import Adam
from keras.layers import Flatten, Dense, Dropout
from keras.callbacks import EarlyStopping, TensorBoard
from keras.models import Model
import random
import pandas as pd
import numpy as np
import tensorflow as tf
import sys
import time
import os
import matplotlib.pyplot as plt

"""
weights: None (random initialization) or imagenet (ImageNet weights). For loading imagenet weights, input_shape should be (331, 331, 3)
"""

def create_dataset(files):
    image_paths = []
    labels = []
    for file in files:
        df = pd.read_csv(file)
        image_paths.extend([p if p.startswith("data/") else os.path.join("data", p) for p in df["file"].tolist()])
        labels.extend(df["year"].tolist())
    return image_paths, labels

def load_and_preprocess_image(path, label, image_size):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, image_size)
    image = image / 255.0
    return image, label

def get_tf_dataset(files, regression=False, class_to_idx=None, min_year=None, max_year=None, image_size=(224,224)):
    image_paths, labels = create_dataset(files)
    image_paths = tf.constant(image_paths)
    if regression:
        # Normalize years to [0, 1]
        labels = [(y - min_year) / (max_year - min_year) for y in labels]
        labels = tf.constant(labels, dtype=tf.float32)
    else:
        labels = [class_to_idx[y] for y in labels]
        labels = tf.constant(labels, dtype=tf.int32)
    ds = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    ds = ds.map(lambda p, l: load_and_preprocess_image(p, l, image_size), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.shuffle(buffer_size=len(image_paths))
    return ds

def build_model(num_classes=None, regression=False, model="InceptionV3", fine_tune=False, input_shape=(224,224,3)):
    base_model = None
    if model == "NASNetMobile":
        base_model = NASNetMobile(include_top=False, weights="imagenet", input_shape=input_shape, pooling="avg")
    elif model == "ResNet101":
        base_model = ResNet101(include_top=False, weights="imagenet", input_shape=input_shape, pooling="avg")
    elif model == "InceptionV3":
        base_model = InceptionV3(include_top=False, weights="imagenet", input_shape=input_shape, pooling="avg")

    x = base_model.output

    if regression:
        predictions = Dense(1, activation="sigmoid", name="PREDICTIONS")(x)
    else:
        predictions = Dense(num_classes, activation="softmax", name="PREDICTIONS")(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    for layer in base_model.layers:
        layer.trainable = False

    # !!--- This is what method says in report ---!!
    # • Fine-tuning One additional layer
    # • Transfer learning One additional layer
    # • Fine-tuning A Dense classifier with a single unit
    # • Transfer learning A Dense classifier with a single unit

    # !!--- If we want to fine tune instead of transfer learning. ---!!
    if fine_tune:
        # Unfreeze one additional layer
        for layer in base_model.layers[-1:]:
            layer.trainable = True

    # ??--- Optional way to do it. ---??

    # freeze_base = True
    # base_model.trainable = not freeze_base

    # inputs = layers.Input(shape=input_shape)
    # x = base_model(inputs, training=not freeze_base)  # if fine-tuning, allow BN layers to train
    # x = layers.GlobalAveragePooling2D()(x)
    return model


def main(task="classification", model="InceptionV3"):
    print(f"Using model: {model}.")
    model_name = model
    start_time = time.time()

    if model == "NASNetMobile":
        image_size = (331, 331)
    elif model == "ResNet101":
        image_size = (224, 224)
    elif model == "InceptionV3":
        image_size = (299, 299)
    else:
        model = "NASNetMobile"
        image_size = (331, 331)
    input_shape = image_size + (3,)

    # --- GPU Configuration for Optimal Utilization ---
    print("TensorFlow Version:", tf.__version__)
    print("Num GPUs Available:", len(tf.config.list_physical_devices("GPU")))
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            # Set memory growth to True to avoid allocating all memory at once
            # tf.config.experimental.set_memory_growth(gpus[0], True)
            tf.config.set_visible_devices(gpus[0], "GPU")
            print("Using GPU:", gpus[0])
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

    if not regression:
        all_years = []
        for file in fold_files:
            df = pd.read_csv(file)
            all_years.extend(df["year"].tolist())
        classes = sorted(list(set(all_years)))
        class_to_idx = {y: i for i, y in enumerate(classes)}
        num_classes = len(classes)
        min_year = None
        max_year = None
    else:
        # For regression, normalize years to [0, 1]
        all_years = []
        for file in fold_files + test_files:
            df = pd.read_csv(file)
            all_years.extend(df["year"].tolist())
        min_year = min(all_years)
        max_year = max(all_years)
        class_to_idx = None
        num_classes = None

    train_ds = get_tf_dataset(
        fold_files, regression=regression, class_to_idx=class_to_idx, min_year=min_year, max_year=max_year, image_size=image_size
    )
    val_ds = get_tf_dataset(
        test_files, regression=regression, class_to_idx=class_to_idx, min_year=min_year, max_year=max_year, image_size=image_size
    )

    # Kanske ta bort dessa? De printar shape mm <ShuffleDataset shapes: ((299, 299, 3), ()), types: (tf.float32, tf.int32)> 
    print(train_ds)
    print(val_ds)

    fine_tune = True
    if fine_tune:
        print("Fine Tuning!")

    model = build_model(num_classes=num_classes, regression=regression, model=model, fine_tune=fine_tune, input_shape=input_shape)

    if regression:
        model.compile(optimizer=Adam(learning_rate=0.0001), loss="mean_squared_error", metrics=["mae", "mse"])
    else:
        model.compile(
            optimizer=Adam(learning_rate=0.0001), loss="sparse_categorical_crossentropy", metrics=["accuracy"]
        )

    # model.summary()

    # --- Callbacks ---
    callbacks = [EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True)]
    log_dir = os.path.join("logs", time.strftime("fit_%Y-%m-%d-%H:%M:%S"))
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
    callbacks.append(tensorboard_callback)

    print(f"TensorBoard logs will be saved to: {log_dir}")
    print("To visualize, run: tensorboard --logdir logs")

    # --- Fit the model ---
    EPOCHS = 100
    BATCH_SIZE = 16

    train_ds_batched = train_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    val_ds_batched = val_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    history = model.fit(train_ds_batched, validation_data=val_ds_batched, epochs=EPOCHS, callbacks=callbacks)

    # --- Evaluate the model ---
    results = model.evaluate(val_ds_batched)
    print("Evaluation results:", results)

    # For regression, denormalize predictions for reporting if needed
    if regression:
        preds = model.predict(val_ds_batched)
        # Denormalize predictions
        preds_years = preds * (max_year - min_year) + min_year
        preds_years_rounded = np.round(preds_years).astype(int)

        # Get true labels from the dataset
        y_true = []
        for _, label in val_ds_batched.unbatch():
            y_true.append(label.numpy())
        y_true = np.array(y_true)
        # Denormalize and round true labels
        y_true_years = y_true * (max_year - min_year) + min_year
        y_true_years_rounded = np.round(y_true_years).astype(int)

        # Count exactly correct predictions
        exact_matches = np.sum(preds_years_rounded.flatten() == y_true_years_rounded.flatten())
        total = len(y_true_years_rounded)
        print(f"Exactly correct year predictions: {exact_matches} out of {total}")

        # Calculate final MAE
        mae = np.mean(np.abs(preds_years_rounded.flatten() - y_true_years_rounded.flatten()))
        print(f"Final MAE (rounded to years): {mae:.2f}")

        print("Sample denormalized predictions (years):", preds_years_rounded[:10].flatten())

    # --- Plot and save training history ---
    # Create plots directory if it doesn't exist
    os.makedirs("plots", exist_ok=True)
    run_id = time.strftime("%Y%m%d-%H%M%S")

    # Plot loss and val_loss
    plt.figure(figsize=(10, 5))
    plt.plot(history.history["loss"], label="loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"plots/loss_val_loss_{model_name}_{run_id}.png")
    plt.close()

    # Plot MAE and MSE if regression
    if regression:
        plt.figure(figsize=(10, 5))
        plt.plot(history.history["mae"], label="mae")
        plt.plot(history.history["mse"], label="mse")
        plt.xlabel("Epoch")
        plt.ylabel("Metric")
        plt.title("Training MAE and MSE")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"plots/train_mae_mse_{model_name}_{run_id}.png")
        plt.close()

        if "val_mae" in history.history and "val_mse" in history.history:
            plt.figure(figsize=(10, 5))
            plt.plot(history.history["val_mae"], label="val_mae")
            plt.plot(history.history["val_mse"], label="val_mse")
            plt.xlabel("Epoch")
            plt.ylabel("Metric")
            plt.title("Validation MAE and MSE")
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"plots/val_mae_mse_{model_name}_{run_id}.png")
            plt.close()

    end_time = time.time()
    running_time = end_time - start_time
    hours = int(running_time // 3600)
    minutes = int((running_time % 3600) // 60)
    seconds = int(running_time % 60)
    print(f"Running time: {hours} hours, {minutes} minutes, {seconds} seconds")


if __name__ == "__main__":
    # Usage: python fashion.py [classification|regression]
    task = sys.argv[1] if len(sys.argv) > 1 else "classification"
    main(task, model="InceptionV3")
