from tensorflow.keras.applications import InceptionV3, ResNet101, NASNetMobile
from tensorflow.keras.optimizers import Adam
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping, TensorBoard, ReduceLROnPlateau
from keras.models import Model
import random
import pandas as pd
import numpy as np
import tensorflow as tf
import sys
import time
import yaml
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc, classification_report


def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def create_dataset(files):
    image_paths = []
    labels = []
    for file in files:
        df = pd.read_csv(file)
        image_paths.extend([p if p.startswith("data/") else os.path.join("data", p) for p in df["file"].tolist()])
        labels.extend(df["year"].tolist())
    return image_paths, labels


data_augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal_and_vertical"),
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomContrast(0.1),
        ]
)


def load_and_preprocess_image(path, label, image_size, augment=False):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, image_size)
    image = image / 255.0
    if augment:
        image = data_augmentation(image)
    return image, label


def get_tf_dataset(
    files, regression=False, class_to_idx=None, min_year=None, max_year=None, augment=False, image_size=(224, 224)
):
    image_paths, labels = create_dataset(files)
    image_paths = tf.constant(image_paths)
    if regression:
        labels = [(y - min_year) / (max_year - min_year) for y in labels]
        labels = tf.constant(labels, dtype=tf.float32)
    else:
        labels = [class_to_idx[y] for y in labels]
        labels = tf.constant(labels, dtype=tf.int32)
    ds = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    ds = ds.map(
        lambda x, y: load_and_preprocess_image(x, y, image_size, augment=augment), num_parallel_calls=tf.data.AUTOTUNE
    )
    ds = ds.shuffle(buffer_size=len(image_paths))
    return ds

def get_input_shape(model_name):
    if model_name == "NASNetMobile":
        return (331, 331, 3)
    elif model_name == "ResNet101":
        return (224, 224, 3)
    elif model_name == "InceptionV3":
        return (299, 299, 3)
    else:
        raise ValueError(f"Unsupported model name: {model_name}")


def train_and_evaluate(train_files, test_file, class_to_idx, num_classes, min_year, max_year, config, fold_idx=None):
    input_shape = get_input_shape(config['model']['name'])
    model_name = config['model']['name']
    regression = config['task'] == 'regression'

    # Prepare datasets
    train_ds = get_tf_dataset(
        train_files,
        regression=regression,
        class_to_idx=class_to_idx,
        min_year=min_year,
        max_year=max_year,
        augment=True,
        image_size=input_shape[:2],
    )

    val_ds = get_tf_dataset(
        [test_file],
        regression=regression,
        class_to_idx=class_to_idx,
        min_year=min_year,
        max_year=max_year,
        augment=False,
        image_size=input_shape[:2],
    )

    BATCH_SIZE = config['training']['batch_size']
    train_ds_batched = train_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    val_ds_batched = val_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    # Build and compile model
    model = build_model(
        config,
        num_classes=num_classes,
        input_shape=input_shape
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=config['training']['learning_rate'])
    loss = 'mean_squared_error' if regression else 'sparse_categorical_crossentropy'
    metrics = ['mae', 'mse'] if regression else ['accuracy']
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    # Callbacks
    callbacks = [EarlyStopping(monitor="val_loss", patience=config['training']['early_stopping_patience'], restore_best_weights=True, verbose=1)]
    log_dir = os.path.join(
        "logs",
        (
            time.strftime(f"fit_fold{fold_idx}_%Y-%m-%d-%H:%M:%S")
            if fold_idx is not None
            else time.strftime("fit_%Y-%m-%d-%H:%M:%S")
        ),
    )
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
    callbacks.append(tensorboard_callback)

    # Train
    history = model.fit(train_ds_batched, validation_data=val_ds_batched, epochs=config['training']['epochs'], callbacks=callbacks, verbose=2)

    # Evaluate
    results = model.evaluate(val_ds_batched)
    print(f"Fold {fold_idx} Evaluation results:", results)

    # Collect predictions and metrics for reporting
    metrics = {}
    if regression:
        preds = model.predict(val_ds_batched)
        preds_years = preds * (max_year - min_year) + min_year
        preds_years_rounded = np.round(preds_years).astype(int)
        y_true = []
        for _, label in val_ds_batched.unbatch():
            y_true.append(label.numpy())
        y_true = np.array(y_true)
        y_true_years = y_true * (max_year - min_year) + min_year
        y_true_years_rounded = np.round(y_true_years).astype(int)
        exact_matches = np.sum(preds_years_rounded.flatten() == y_true_years_rounded.flatten())
        total = len(y_true_years_rounded)
        mae = np.mean(np.abs(preds_years_rounded.flatten() - y_true_years_rounded.flatten()))
        print(f"Fold {fold_idx} Exactly correct year predictions: {exact_matches} out of {total}")
        print(f"Fold {fold_idx} Final MAE (rounded to years): {mae:.2f}")
        metrics = {"mae": mae, "exact": exact_matches, "total": total}

        run_id = time.strftime("%Y%m%d-%H%M%S")
        os.makedirs("plots", exist_ok=True)
        plt.figure(figsize=(10, 5))
        plt.plot(history.history["loss"], label="loss")
        plt.plot(history.history["val_loss"], label="val_loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"Regression: Training and Validation Loss (Fold {fold_idx})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"plots/loss_val_loss_regression_{model_name}_fold{fold_idx}_{run_id}.png")
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
            plt.savefig(f"plots/train_mae_mse_{model_name}_fold{fold_idx}_{run_id}.png")
            plt.close()

    else:
        y_true = []
        y_pred = []
        for images, labels in val_ds_batched:
            preds = model.predict(images)
            y_true.extend(labels.numpy())
            y_pred.extend(np.argmax(preds, axis=1))

            # Print classification report
            target_names = [str(k) for k in sorted(class_to_idx.keys(), key=lambda x: class_to_idx[x])]
            report = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)
            print(classification_report(y_true, y_pred, target_names=target_names))

            # Plot bar chart of per-class F1-score
            f1_scores = [report[str(i)]["f1-score"] for i in range(len(target_names))]
            plt.figure(figsize=(10, 6))
            plt.bar(target_names, f1_scores, color="skyblue")
            plt.xlabel("Class")
            plt.ylabel("F1-score")
            plt.title(f"Per-Class F1-score (Fold {fold_idx})")
            plt.ylim(0, 1)
            plt.tight_layout()
            plt.savefig(f"plots/f1_score_bar_fold{fold_idx}.png")
            plt.close()

        if class_to_idx is not None:
            idx_to_class = {v: k for k, v in class_to_idx.items()}
            y_true_years = np.array([idx_to_class[idx] for idx in y_true])
            y_pred_years = np.array([idx_to_class[idx] for idx in y_pred])
            mae_years = np.mean(np.abs(y_true_years - y_pred_years))
            print(f"Classification MAE (in years): {mae_years:.2f}")

        # Binarize labels for ROC computation
        y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))

        # Collect predictions
        y_score = []
        for images, _ in val_ds_batched:
            preds = model.predict(images)
            y_score.extend(preds)
        y_score = np.array(y_score)

        # Compute micro-average ROC
        fpr_micro, tpr_micro, _ = roc_curve(y_true_bin.ravel(), y_score.ravel())
        roc_auc_micro = auc(fpr_micro, tpr_micro)

        # Plot micro-average ROC
        plt.figure(figsize=(8, 6))
        plt.plot(fpr_micro, tpr_micro, color="blue", lw=2, label=f"Micro-average ROC (AUC = {roc_auc_micro:.2f})")
        plt.plot([0, 1], [0, 1], "k--", lw=1)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"Micro-average ROC Curve (Fold {fold_idx})")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(f"plots/micro_avg_roc_curve_fold{fold_idx}.png")
        plt.close()

        # Return metrics
        metrics = {"mae": results[0], "accuracy": results[1]}
    return metrics


def build_model(config, num_classes=None, input_shape=(224, 224, 3)):
    
    regression = config['task'] == 'regression'
    model_name = config['model']['name']
    fine_tune = config['model']['fine_tune']

    base_model = None
    if model_name == "NASNetMobile":
        base_model = NASNetMobile(include_top=False, weights="imagenet", input_shape=input_shape, pooling="avg")
    elif model_name == "ResNet101":
        base_model = ResNet101(include_top=False, weights="imagenet", input_shape=input_shape, pooling="avg")
    elif model_name == "InceptionV3":
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


def main():
    config = load_config("config.yaml")
    regression = config['task'] == 'regression'
    model_name = config['model']['name']
    fine_tune = config['model']['fine_tune']
    fold_nums = [num for num in range(10)]
    input_shape = get_input_shape(config['model']['name'])
    print(f"Using model: {model_name}.")

    # --- GPU Configuration for Optimal Utilization ---
    print("TensorFlow Version:", tf.__version__)
    print("Num GPUs Available:", len(tf.config.list_physical_devices("GPU")))
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            tf.config.set_visible_devices(gpus[0], "GPU")
            print("Using GPU:", gpus[0])
        except RuntimeError as e:
            print(e)
    else:
        print("No GPU detected. Training will run on CPU.")

    start_time = time.time()

    if config['cross_validation']:
        all_metrics = []
        for test_fold in fold_nums:
            print(f"\n=== Fold {test_fold} ===")
            train_folds = [fold for fold in fold_nums if fold != test_fold]
            train_files = [f"data/datasets/fold{fold}.csv" for fold in train_folds]
            test_file = f"data/datasets/fold{test_fold}.csv"

            if not regression:
                all_years = []
                for file in train_files:
                    df = pd.read_csv(file)
                    all_years.extend(df["year"].tolist())
                classes = sorted(list(set(all_years)))
                class_to_idx = {y: i for i, y in enumerate(classes)}
                num_classes = len(classes)
                min_year = None
                max_year = None
            else:
                all_years = []
                for file in train_files + [test_file]:
                    df = pd.read_csv(file)
                    all_years.extend(df["year"].tolist())
                min_year = min(all_years)
                max_year = max(all_years)
                class_to_idx = None
                num_classes = None

            metrics = train_and_evaluate(
                train_files,
                test_file,
                class_to_idx,
                num_classes,
                min_year,
                max_year,
                config,
                fold_idx=None,
            )
            all_metrics.append(metrics)

        # Print mean metrics
        if regression:
            maes = [m["mae"] for m in all_metrics]
            exacts = [m["exact"] for m in all_metrics]
            totals = [m["total"] for m in all_metrics]
            print(f"\nMean MAE over 10 folds: {np.mean(maes):.2f}")
            print(f"Mean exact matches: {np.mean(exacts):.2f} / {np.mean(totals):.2f}")
        else:
            accs = [m["accuracy"] for m in all_metrics]
            print(f"\nMean accuracy over 10 folds: {np.mean(accs):.4f}")

    else:
        # Single train/test split (random fold selection)
        test_fold = random.choice(fold_nums)
        print(f"Test fold: {test_fold}")
        train_folds = [fold for fold in fold_nums if fold != test_fold]
        train_files = [f"data/datasets/fold{fold}.csv" for fold in train_folds]
        test_file = f"data/datasets/fold{test_fold}.csv"

        if not regression:
            all_years = []
            for file in train_files:
                df = pd.read_csv(file)
                all_years.extend(df["year"].tolist())
            classes = sorted(list(set(all_years)))
            class_to_idx = {y: i for i, y in enumerate(classes)}
            num_classes = len(classes)
            min_year = None
            max_year = None
        else:
            all_years = []
            for file in train_files + [test_file]:
                df = pd.read_csv(file)
                all_years.extend(df["year"].tolist())
            min_year = min(all_years)
            max_year = max(all_years)
            class_to_idx = None
            num_classes = None

        metrics = train_and_evaluate(
            train_files,
            test_file,
            class_to_idx,
            num_classes,
            min_year,
            max_year,
            config,
            fold_idx=None,
        )
        print("Metrics:", metrics)

    end_time = time.time()
    running_time = end_time - start_time
    hours = int(running_time // 3600)
    minutes = int((running_time % 3600) // 60)
    seconds = int(running_time % 60)
    print(f"Total running time: {hours} hours, {minutes} minutes, {seconds} seconds")


if __name__ == "__main__":
    main()
