from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
)
from sklearn.utils.class_weight import compute_class_weight
from tuning import run_hyperparameter_tuning, apply_best_hyperparameters, get_input_shape
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
import pandas as pd
import numpy as np
import time
import os
import yaml


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


data_augmentation = tf.keras.models.Sequential(
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


# def get_tf_dataset(
#     files, regression=False, class_to_idx=None, min_year=None, max_year=None, augment=False, image_size=(224, 224)
# ):
#     image_paths, labels = create_dataset(files)
#     image_paths = tf.constant(image_paths)
#     if regression:
#         labels = [(y - min_year) / (max_year - min_year) for y in labels]
#         labels = tf.constant(labels, dtype=tf.float32)
#     else:
#         labels = [class_to_idx[y] for y in labels]
#         labels = tf.constant(labels, dtype=tf.int32)
#     ds = tf.data.Dataset.from_tensor_slices((image_paths, labels))
#     ds = ds.map(
#         lambda x, y: load_and_preprocess_image(x, y, image_size, augment=augment), num_parallel_calls=tf.data.AUTOTUNE
#     )
#     ds = ds.shuffle(buffer_size=len(image_paths))
#     return ds


def get_tf_dataset(
    files, regression=False, class_to_idx=None, min_year=None, max_year=None, augment=False, image_size=(224, 224)
):
    image_paths, labels = create_dataset(files)
    image_paths = tf.constant(image_paths)

    if regression:
        labels = [(y - min_year) / (max_year - min_year) for y in labels]
        labels = tf.constant(labels, dtype=tf.float32)
    else:
        # First convert to integer indices
        labels_idx = [class_to_idx[y] for y in labels]
        # Then convert to one-hot encoding
        labels = tf.one_hot(indices=labels_idx, depth=len(class_to_idx))

    ds = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    ds = ds.map(
        lambda x, y: load_and_preprocess_image(x, y, image_size, augment=augment), num_parallel_calls=tf.data.AUTOTUNE
    )
    ds = ds.shuffle(buffer_size=len(image_paths))
    return ds


def get_input_shape(model_name):
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


def get_class_weights(labels, method="balanced", max_weight=0.75):
    classes = np.array(sorted(set(labels)))
    weights = compute_class_weight(class_weight=method, classes=classes, y=labels)

    capped_weights = {cls: min(w, max_weight) for cls, w in zip(classes, weights)}
    return capped_weights


def plot_class_distribution(train_labels, class_weights=None, plot_dir=None):
    plt.figure(figsize=(12, 6))

    # Count occurrences of each class
    unique_labels = sorted(set(train_labels))
    counts = [train_labels.count(label) for label in unique_labels]

    # Create bar chart of class distribution
    ax = plt.subplot(1, 2, 1)
    ax.bar(range(len(unique_labels)), counts)
    ax.set_title("Class Distribution")
    ax.set_xlabel("Class Index")
    ax.set_ylabel("Count")
    ax.set_xticks(range(len(unique_labels)), unique_labels)
    step = max(1, len(unique_labels) // 10)
    ax.set_xticklabels([str(label) if i % step == 0 else "" for i, label in enumerate(unique_labels)])

    ax = plt.subplot(1, 2, 2)
    weights = [class_weights.get(label, 0) for label in unique_labels]
    ax.bar(range(len(unique_labels)), weights)
    ax.set_title("Class Weights")
    ax.set_xlabel("Class Index")
    ax.set_ylabel("Weight")
    ax.set_xticks(range(len(unique_labels)))
    step = max(1, len(unique_labels) // 10)
    ax.set_xticklabels([str(label) if i % step == 0 else "" for i, label in enumerate(unique_labels)])

    plt.tight_layout()

    if plot_dir:
        os.makedirs(plot_dir, exist_ok=True)
        plt.savefig(os.path.join(plot_dir, "class_distribution_weights.png"))
    plt.close()


def top_n_accuracy(y_true, y_score, n=3):
    top_n = np.argsort(y_score, axis=1)[:, -n:]
    return np.mean([y in top_n_row for y, top_n_row in zip(y_true, top_n)])


def plot_cv_metrics_summary(all_metrics, plot_dir):
    if not all_metrics or not isinstance(all_metrics, list):
        print("No metrics to plot.")
        return

    # Collect all metric keys
    metric_keys = all_metrics[0].keys()
    stats = {}

    for key in metric_keys:
        values = [m[key] for m in all_metrics if key in m]
        stats[key] = {
            "mean": np.mean(values),
            "std": np.std(values),
            "var": np.var(values),
            "all": values,
        }

    # Plot
    plt.figure(figsize=(8, 6))
    x = np.arange(len(metric_keys))
    means = [stats[k]["mean"] for k in metric_keys]
    stds = [stats[k]["std"] for k in metric_keys]
    vars_ = [stats[k]["var"] for k in metric_keys]

    plt.bar(x, means, yerr=stds, capsize=8, color="green", label="Mean ± Std")
    plt.scatter(x, means, color="blue")
    plt.xticks(x, metric_keys)
    plt.ylabel("Metric Value")
    plt.title("Cross-Validation Metrics Summary (Mean ± Std)")
    plt.tight_layout()
    plt.legend()

    # Annotate variance
    for i, v in enumerate(vars_):
        plt.text(i, means[i] + stds[i] + 0.01, f"Var: {v:.4f}", ha="center", fontsize=9)

    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(os.path.join(plot_dir, "cv_metrics_summary.png"))
    plt.close()


def plot_metrics(
    history=None, fold_str="", plot_dir="", regression=False, class_to_idx=None, y_true=None, y_pred=None, y_score=None
):
    if regression:
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

        mcc = matthews_corrcoef(y_true, y_pred)
        print(f"Matthews Correlation Coefficient: {mcc:.3f}")

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

    return


def ordinal_categorical_cross_entropy(y_true, y_pred):
    num_classes = tf.cast(tf.shape(y_pred)[-1], tf.float32)
    true_labels = tf.argmax(y_true, axis=-1)
    pred_labels = tf.argmax(y_pred, axis=-1)
    weights = tf.abs(tf.cast(pred_labels - true_labels, tf.float32)) / (num_classes - 1.0)
    base_loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
    loss = (1.0 + weights) * base_loss
    return loss


def train_and_evaluate(train_files, test_file, class_to_idx, num_classes, min_year, max_year, config, run_id, fold_idx):
    input_shape = get_input_shape(config.get("model", {}).get("name", "InceptionV3"))
    model_name = config.get("model", {}).get("name", "InceptionV3")
    regression = config.get("task", "classification") == "regression"
    metrics = {}

    # Helper for fold-specific naming
    fold_str = f"_fold{fold_idx}" if fold_idx is not None else ""
    fold_print = f"Fold {fold_idx} " if fold_idx is not None else ""

    # Set up plot and log directories
    if config.get("cross_validation", False) and fold_idx is not None:
        plot_dir = os.path.join("plots", "10_fold_cv", f"{run_id}_{model_name}_fold{fold_idx}")
        log_dir = os.path.join("logs", "tensorboard", "10_fold_cv", f"{run_id}_fit_fold{fold_idx}")
    else:
        plot_dir = os.path.join("plots", "single_run", f"{run_id}_{model_name}_fold{fold_idx}")
        log_dir = os.path.join("logs", "tensorboard", f"fit_{run_id}_{model_name}_fit_fold{fold_idx}")

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

    # Calculate class weights for classification tasks
    class_weights = None
    if not regression and config.get("training", {}).get("class_balancing", {}).get("enabled", True):
        method = config.get("training", {}).get("class_balancing", {}).get("method", "balanced")
        # Extract labels as before
        train_labels = []
        for file in train_files:
            df = pd.read_csv(file)
            train_labels.extend([class_to_idx[y] for y in df["year"].tolist()])

        class_weights = get_class_weights(train_labels, method)
        print(f"{fold_print}Class weights: {class_weights}")
        if class_weights:
            plot_class_distribution(train_labels, class_weights, plot_dir)

    # Run hyperparameter tuning if enabled
    if config.get("hyperparameter_tuning", {}).get("enabled", False):
        print(f"{fold_print}Running hyperparameter tuning...")
        best_params = run_hyperparameter_tuning(train_ds, val_ds, config, num_classes, run_id, fold_idx)
        # Apply the best parameters to the config
        config = apply_best_hyperparameters(config, best_params)
        print(
            f"{fold_print}Using tuned parameters: batch_size={config['training']['batch_size']}, "
            + f"dropout={config['model']['dropout']}, lr={config['training']['learning_rate']}"
        )

        # Common code for both paths - prepare dataset with batching
        BATCH_SIZE = config["training"]["batch_size"]
        train_ds_batched = train_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
        val_ds_batched = val_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

        # Build and compile model
        model = build_model(
            config,
            num_classes=num_classes,
            input_shape=input_shape,
        )

        optimizer = tf.keras.optimizers.Adam(learning_rate=config["training"]["learning_rate"])
        # loss = "mean_squared_error" if regression else ordinal_categorical_cross_entropy
        loss = tf.keras.losses.CategoricalFocalCrossentropy(
            alpha=0.25,  # Can be a scalar or a list with per-class weights
            gamma=2.0,  # Controls focus on hard examples (higher = more focus)
        )
        metrics_list = ["mae", "mse"] if regression else ["accuracy"]
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics_list)

        # Callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=config["training"]["early_stopping_patience"],
                restore_best_weights=True,
                min_delta=1e-4,
                verbose=1,
            )
        ]

        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        callbacks.append(tensorboard_callback)

        reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=config["training"]["reduce_lr_patience"],
            min_delta=1e-4,
            verbose=1,
            min_lr=1e-6,
        )

        callbacks.append(reduce_lr_callback)

        if config["model"]["save_model"]:
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
            class_weight=class_weights,
        )

        # Evaluate
        results = model.evaluate(val_ds_batched, verbose=0)
        print(f"{fold_print}Evaluation results:", results)

        # Collect predictions and metrics for reporting
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

            plot_metrics(history, fold_str, plot_dir, regression, class_to_idx)

    else:
        BATCH_SIZE = config["training"]["batch_size"]
        train_ds_batched = train_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
        val_ds_batched = val_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

        # Build and compile model
        model = build_model(
            config,
            num_classes=num_classes,
            input_shape=input_shape,
        )

        optimizer = tf.keras.optimizers.Adam(learning_rate=config["training"]["learning_rate"])
        # loss = "mean_squared_error" if regression else ordinal_categorical_cross_entropy
        loss = tf.keras.losses.CategoricalFocalCrossentropy(
            alpha=0.25,  # Can be a scalar or a list with per-class weights
            gamma=2.0,  # Controls focus on hard examples (higher = more focus)
        )
        metrics = ["mae", "mse"] if regression else ["accuracy"]
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        # Callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=config["training"]["early_stopping_patience"],
                restore_best_weights=True,
                min_delta=1e-4,
                verbose=1,
            )
        ]

        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        callbacks.append(tensorboard_callback)

        reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=config["training"]["reduce_lr_patience"],
            min_delta=1e-4,
            verbose=1,
            min_lr=1e-6,
        )

        callbacks.append(reduce_lr_callback)

        # model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
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
            class_weight=class_weights,
        )

        # Evaluate
        results = model.evaluate(val_ds_batched, verbose=0)
        print(f"{fold_print}Evaluation results:", results)

        # Collect predictions and metrics for reporting
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

            plot_metrics(history, fold_str, plot_dir, regression, class_to_idx)

        else:
            # Collect predictions
            y_score = []
            y_true = []
            y_pred = []
            for images, labels in val_ds_batched:
                preds = model.predict(images, verbose=0)
                y_true.extend(labels.numpy())
                y_pred.extend(np.argmax(preds, axis=1))
                y_score.extend(preds)
            y_score = np.array(y_score)

            mcc = matthews_corrcoef(y_true, y_pred)

            plot_metrics(history, fold_str, plot_dir, regression, class_to_idx, y_true, y_pred, y_score)

            if class_to_idx is not None:
                idx_to_class = {v: k for k, v in class_to_idx.items()}
                y_true_years = np.array([idx_to_class[idx] for idx in y_true])
                y_pred_years = np.array([idx_to_class[idx] for idx in y_pred])
                mae_years = np.mean(np.abs(y_true_years - y_pred_years))
                print(f"Classification MAE (in years): {mae_years:.2f}")

            metrics = {"accuracy": results[1], "mae_years": mae_years, "mcc": mcc}
    return metrics


def build_model(config, num_classes=None, input_shape=(224, 224, 3)):
    regression = config.get("task", "classification") == "regression"
    model_name = config.get("model", {}).get("name", "InceptionV3")
    fine_tune = config["model"]["fine_tune"].get("use", False)
    fine_tune_layers = config["model"]["fine_tune"].get("layers", 1)
    dense_units = config["model"].get("dense_units", 1)
    dense_layers = config["model"].get("dense_layers", 1)

    # Select base model and setup according to model_name
    if model_name == "NASNetMobile":
        base_model = tf.keras.applications.NASNetMobile(weights="imagenet", input_shape=input_shape, include_top=False)
        base_model.trainable = False
        x = base_model.output
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(76, activation="relu")(x)
    elif model_name == "ResNet101":
        base_model = tf.keras.applications.ResNet101(weights="imagenet", input_shape=input_shape, include_top=False)
        base_model.trainable = False
        x = base_model.output
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(10, activation="relu")(x)
    elif model_name == "InceptionV3":
        base_model = tf.keras.applications.InceptionV3(weights="imagenet", input_shape=input_shape, include_top=False)
        base_model.trainable = False
        x = base_model.output
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
    elif model_name == "EfficientNetB3":
        base_model = tf.keras.applications.EfficientNetB3(
            weights="imagenet", input_shape=input_shape, include_top=False
        )
        base_model.trainable = False
        x = base_model.output
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
    else:
        raise ValueError(f"Unsupported model name: {model_name}")

    # Add dense layers based on config
    for _ in range(dense_layers):
        x = tf.keras.layers.Dense(dense_units, activation="relu")(x)
        # Optional dropout after each dense layer
        if config["model"].get("dropout", {}).get("use", False):
            x = tf.keras.layers.Dropout(config["model"].get("dropout", {}).get("value", 0))(x)

    # Output layer
    if regression:
        if config["model"].get("l2_regularization", {}).get("use", False):
            predictions = tf.keras.layers.Dense(
                1,
                activation="sigmoid",
                name="PREDICTIONS",
                kernel_regularizer=tf.keras.regularizers.l2(
                    float(config["model"].get("l2_regularization", {}).get("value", 0))
                ),
            )(x)
        else:
            predictions = tf.keras.layers.Dense(1, activation="sigmoid", name="PREDICTIONS")(x)
    else:
        if config["model"].get("l2_regularization", {}).get("use", False):
            predictions = tf.keras.layers.Dense(
                num_classes,
                activation="softmax",
                name="PREDICTIONS",
                kernel_regularizer=tf.keras.regularizers.l2(
                    float(config["model"].get("l2_regularization", {}).get("value", 0))
                ),
            )(x)
        else:
            predictions = tf.keras.layers.Dense(num_classes, activation="softmax", name="PREDICTIONS")(x)

    model = tf.keras.models.Model(inputs=base_model.input, outputs=predictions)

    # Fine-tuning: unfreeze specified number of layers if requested
    if fine_tune:
        print(f"\n===== Fine Tuning {fine_tune_layers} layers! =====")
        for layer in base_model.layers[-fine_tune_layers:]:
            layer.trainable = True

    return model


def main():
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

    config = load_config("config.yaml")
    regression = regression = config.get("task", "classification") == "regression"
    model_name = config.get("model", {}).get("name", "InceptionV3")
    fold_nums = [num for num in range(10)]
    run_id = time.strftime("%Y-%m-%d_%H:%M:%S")
    print(f"\n=== Using model: {model_name}. ===")
    print(f"RUN ID: {run_id}")

    if config["cross_validation"]:
        all_metrics = []
        for test_fold in fold_nums:
            print(f"\n===== Fold {test_fold} =====")
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
                run_id,
                test_fold,
            )
            all_metrics.append(metrics)
        plot_cv_metrics_summary(
            all_metrics, plot_dir=os.path.join("plots", "10_fold_cv", f"{run_id}_{model_name}_mean_std_var")
        )

        # Print mean metrics
        if regression:
            maes = [m["mae"] for m in all_metrics]
            exacts = [m["exact"] for m in all_metrics]
            totals = [m["total"] for m in all_metrics]
            print(f"\nMean MAE over 10 folds: {np.mean(maes):.2f}")
            print(f"Mean exact matches: {np.mean(exacts):.2f} / {np.mean(totals):.2f}")
        else:
            accs = [m["accuracy"] for m in all_metrics]
            maes = [m["mae_years"] for m in all_metrics]
            mccs = [m["mcc"] for m in all_metrics]
            print(f"\nMean MAE over 10 folds: {np.mean(maes):.4f}")
            print(f"\nMean accuracy over 10 folds: {np.mean(accs):.4f}")
            print(f"\nMean Matthews Correlation Coefficient over 10 folds: {np.mean(mccs):.4f}")

    else:
        # Single train/test split (random fold selection)
        test_fold = fold_nums[0]
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
            run_id,
            test_fold,
        )
        print("Metrics:", metrics)

    end_time = time.time()
    running_time = end_time - start_time
    hours = int(running_time // 3600)
    minutes = int((running_time % 3600) // 60)
    seconds = int(running_time % 60)
    print(f"\n=== Total running time: {hours} hours, {minutes} minutes, {seconds} seconds ===\n")


if __name__ == "__main__":
    main()
