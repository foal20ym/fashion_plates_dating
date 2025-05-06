from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, RandomRotation, RandomFlip, RandomContrast
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import InceptionV3, ResNet101, NASNetMobile
from tensorflow.keras.regularizers import l2
import random
import pandas as pd
import numpy as np
import tensorflow as tf
import time
import os
import yaml
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, roc_curve, classification_report, confusion_matrix, f1_score
import seaborn as sns


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


data_augmentation = Sequential(
    [
        RandomFlip("horizontal_and_vertical"),
        RandomRotation(0.1),
        RandomContrast(0.1),
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


def get_input_shape(model_name, include_top):
    if model_name == "NASNetMobile":
        return (224, 224, 3)
    elif model_name == "ResNet101":
        return (224, 224, 3)
    elif model_name == "InceptionV3":
        return (299, 299, 3)
    else:
        raise ValueError(f"Unsupported model name: {model_name}")


def get_highest_version_for_saved_model(model_name):
    model_dir = "trained_models"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        return 1

    model_files = [f for f in os.listdir(model_dir) if f.startswith(f"{model_name}_version_") and f.endswith(".keras")]

    if not model_files:
        return 1

    versions = []
    for file in model_files:
        try:
            version = int(file.split("_version_")[-1].split(".")[0])
            versions.append(version)
        except (ValueError, IndexError):
            continue

    return max(versions) + 1 if versions else 1


def top_n_accuracy(y_true, y_score, n=3):
    top_n = np.argsort(y_score, axis=1)[:, -n:]
    return np.mean([y in top_n_row for y, top_n_row in zip(y_true, top_n)])


def train_and_evaluate(train_files, test_file, class_to_idx, num_classes, min_year, max_year, config, fold_idx=None):
    input_shape = get_input_shape(config["model"]["name"], config["model"]["include_top"])
    model_name = config["model"]["name"]
    regression = config["task"] == "regression"

    # Helper for fold-specific naming
    fold_str = f"_fold{fold_idx}" if fold_idx is not None else ""
    fold_print = f"Fold {fold_idx} " if fold_idx is not None else ""
    run_id = time.strftime("%Y-%m-%d_%H:%M:%S")

    # Set up plot and log directories
    if config.get("cross_validation", False) and fold_idx is not None:
        plot_dir = os.path.join("plots", "10_fold_cv", f"{run_id}_{model_name}_fold{fold_idx}")
        log_dir = os.path.join("logs", "tensorboard", "10_fold_cv", f"{run_id}_fit_fold{fold_idx}")
    else:
        plot_dir = "plots"
        log_dir = os.path.join("logs", "tensorboard", f"fit_{run_id}")

    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

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

    BATCH_SIZE = config["training"]["batch_size"]
    train_ds_batched = train_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    val_ds_batched = val_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    # Build and compile model
    model = build_model(
        config,
        num_classes=num_classes,
        input_shape=input_shape,
    )

    optimizer = Adam(learning_rate=config["training"]["learning_rate"])
    loss = "mean_squared_error" if regression else "sparse_categorical_crossentropy"
    metrics = ["mae", "mse"] if regression else ["accuracy"]
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=config["training"]["early_stopping_patience"],
            restore_best_weights=True,
            min_delta=1e-4,
            verbose=1,
        )
    ]

    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
    callbacks.append(tensorboard_callback)

    reduce_lr_callback = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.1,
        patience=config["training"]["reduce_lr_patience"],
        min_delta=1e-4,
        verbose=1,
        min_lr=1e-6,
    )

    callbacks.append(reduce_lr_callback)

    # model_checkpoint = ModelCheckpoint(
    #     filepath,
    #     monitor="val_loss",
    #     verbose=1,
    #     save_best_only=True,
    #     save_weights_only=False,
    #     mode="auto",
    #     save_freq="epoch",
    #     initial_value_threshold=None,
    # )

    if config["model"]["save_model"]:
        # callbacks.append(model_checkpoint)
        if not os.path.exists("trained_models"):
            os.makedirs("trained_models")
        version = get_highest_version_for_saved_model(model_name)
        model.save(f"trained_models/{model_name}_version_{version}.keras")

    # Train
    history = model.fit(
        train_ds_batched,
        validation_data=val_ds_batched,
        epochs=config["training"]["epochs"],
        callbacks=callbacks,
        verbose=2,
    )

    # Evaluate
    results = model.evaluate(val_ds_batched, verbose=0)
    print(f"{fold_print}Evaluation results:", results)

    # Collect predictions and metrics for reporting
    metrics = {}
    if regression:
        preds = model.predict(val_ds_batched, verbose=0)
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
        print(f"{fold_print}Exactly correct year predictions: {exact_matches} out of {total}")
        print(f"{fold_print}Final MAE (rounded to years): {mae:.2f}")
        metrics = {"mae": mae, "exact": exact_matches, "total": total}

        os.makedirs("plots", exist_ok=True)
        plt.figure(figsize=(10, 5))
        plt.plot(history.history["loss"], label="loss")
        plt.plot(history.history["val_loss"], label="val_loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"Regression: Training and Validation Loss{fold_str}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"loss_val_loss_regression.png"))
        plt.close()

        if "val_mae" in history.history and "val_mse" in history.history:
            plt.figure(figsize=(10, 5))
            plt.plot(history.history["val_mae"], label="val_mae")
            plt.plot(history.history["val_mse"], label="val_mse")
            plt.xlabel("Epoch")
            plt.ylabel("Metric")
            plt.title(f"Validation MAE and MSE{fold_str}")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, f"train_mae_mse.png"))
            plt.close()

    else:
        # Collect predictions
        y_score = []
        y_true = []
        y_pred = []
        for images, labels in val_ds_batched:
            preds = model.predict(images, verbose=2)
            y_true.extend(labels.numpy())
            y_pred.extend(np.argmax(preds, axis=1))
            y_score.extend(preds)
        y_score = np.array(y_score)

        target_names = [str(k) for k in sorted(class_to_idx.keys(), key=lambda x: class_to_idx[x])]
        all_labels = [class_to_idx[k] for k in sorted(class_to_idx.keys(), key=lambda x: class_to_idx[x])]

        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred, labels=all_labels)
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=False, fmt="d", cmap="Blues", xticklabels=target_names, yticklabels=target_names)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(f"Confusion Matrix{fold_str}")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"confusion_matrix.png"))
        plt.close()

        # Print classification report
        report = classification_report(
            y_true, y_pred, labels=all_labels, target_names=target_names, output_dict=True, zero_division=0
        )
        print(classification_report(y_true, y_pred, labels=all_labels, target_names=target_names, zero_division=0))

        print(f"Macro avg F1: {report['macro avg']['f1-score']:.3f}")
        print(f"Weighted avg F1: {report['weighted avg']['f1-score']:.3f}")
        micro_f1 = f1_score(y_true, y_pred, average="micro")
        print(f"Micro avg F1: {micro_f1:.3f}")

        top3_acc = top_n_accuracy(y_true, y_score, n=3)
        top5_acc = top_n_accuracy(y_true, y_score, n=5)
        print(f"Top-3 Accuracy: {top3_acc:.3f}")
        print(f"Top-5 Accuracy: {top5_acc:.3f}")

        # Plot bar chart of per-class F1-score
        f1_scores = [report[name]["f1-score"] for name in target_names]
        plt.figure(figsize=(max(10, len(target_names) * 0.5), 6))
        bar_positions = np.arange(len(target_names))
        plt.bar(bar_positions, f1_scores, color="green", width=0.6, align="center")
        plt.xticks(bar_positions, target_names, rotation=45, ha="center")
        plt.xlabel("Class")
        plt.ylabel("F1-score")
        plt.title(f"Per-Class F1-score{fold_str}")
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"f1_score_bar.png"))
        plt.close()

        if class_to_idx is not None:
            idx_to_class = {v: k for k, v in class_to_idx.items()}
            y_true_years = np.array([idx_to_class[idx] for idx in y_true])
            y_pred_years = np.array([idx_to_class[idx] for idx in y_pred])
            mae_years = np.mean(np.abs(y_true_years - y_pred_years))
            print(f"Classification MAE (in years): {mae_years:.2f}")

        y_true_bin = label_binarize(y_true, classes=all_labels)  # shape (n_samples, n_classes)

        # Find classes present in y_true
        present_classes = [i for i, label in enumerate(all_labels) if label in y_true]

        # Compute scalar AUCs with roc_auc_score
        roc_auc_micro = roc_auc_score(y_true_bin, y_score, average="micro", multi_class="ovr")
        print(f"Micro ROC AUC  = {roc_auc_micro:.2f}")
        # Compute macro AUC only for present classes
        if len(present_classes) > 1:
            roc_auc_macro = roc_auc_score(
                label_binarize(y_true, classes=[all_labels[i] for i in present_classes]),
                y_score[:, present_classes],
                average="macro",
                multi_class="ovr",
            )
            print(f"Macro ROC AUC (present classes) = {roc_auc_macro:.2f}")
        else:
            print("Macro ROC AUC not defined (less than 2 classes present in y_true)")

        # Build the ROC curves for plotting

        # Micro-average ROC curve
        fpr_micro, tpr_micro, _ = roc_curve(y_true_bin.ravel(), y_score.ravel())

        # Macro-average ROC curve (average of per-class curves)
        fpr_dict = {}
        tpr_dict = {}
        for i in range(len(all_labels)):
            fpr_dict[i], tpr_dict[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])

        all_fpr = np.unique(np.concatenate([fpr_dict[i] for i in range(len(all_labels))]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(len(all_labels)):
            mean_tpr += np.interp(all_fpr, fpr_dict[i], tpr_dict[i])
        mean_tpr /= len(all_labels)

        # Plot both curves
        plt.figure(figsize=(8, 6))
        plt.plot(fpr_micro, tpr_micro, color="blue", lw=2, label=f"Micro-average ROC (AUC = {roc_auc_micro:.2f})")
        plt.plot(all_fpr, mean_tpr, color="green", lw=2, label=f"Macro-average ROC (AUC = {roc_auc_macro:.2f})")
        plt.plot([0, 1], [0, 1], "k--", lw=1)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"Micro & Macro-average ROC Curve{fold_str}")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"micro_macro_avg_roc_curve.png"))
        plt.close()

        # Return metrics
        metrics = {"mae": results[0], "accuracy": results[1]}
    return metrics


def build_model(config, num_classes=None, input_shape=(224, 224, 3)):
    regression = config["task"] == "regression"
    model_name = config["model"]["name"]
    fine_tune = config["model"]["fine_tune"]
    print("model_name:", model_name)
    print("input_shape:", input_shape)

    base_model = None
    if model_name == "NASNetMobile":
        base_model = NASNetMobile(
            include_top=config["model"]["include_top"], weights="imagenet", input_shape=input_shape, pooling="avg"
        )
    elif model_name == "ResNet101":
        base_model = ResNet101(
            include_top=config["model"]["include_top"], weights="imagenet", input_shape=input_shape, pooling="avg"
        )
    elif model_name == "InceptionV3":
        base_model = InceptionV3(
            include_top=config["model"]["include_top"], weights="imagenet", input_shape=input_shape, pooling="avg"
        )

    x = base_model.output
    if config["model"]["use_dropout"]:
        x = Dropout(config["model"]["dropout"])(x)

    if regression:
        if config["model"]["use_l2_regularization"]:
            predictions = Dense(
                1, activation="sigmoid", name="PREDICTIONS", kernel_regularizer=l2(config["model"]["l2_regularization"])
            )(x)
        else:
            predictions = Dense(1, activation="sigmoid", name="PREDICTIONS")(x)
    else:
        if config["model"]["use_l2_regularization"]:
            predictions = Dense(
                num_classes,
                activation="softmax",
                name="PREDICTIONS",
                kernel_regularizer=l2(config["model"]["l2_regularization"]),
            )(x)
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
    regression = config["task"] == "regression"
    model_name = config["model"]["name"]
    fold_nums = [num for num in range(10)]
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

    if config["cross_validation"]:
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
                fold_idx=test_fold,
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
            fold_idx=test_fold,
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
