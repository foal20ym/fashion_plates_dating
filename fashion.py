from sklearn.metrics import (
    matthews_corrcoef,
)
from sklearn.utils.class_weight import compute_class_weight
from tuning import run_hyperparameter_tuning, apply_best_hyperparameters, get_input_shape
import tensorflow as tf
import pandas as pd
import numpy as np
import time
import os
import yaml
from plotting import plot_cv_metrics_summary, plot_metrics, plot_class_distribution
import keras_cv  # Had to run: pip install --upgrade keras-cv-nightly tf-nightly
from keras.saving import register_keras_serializable


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


rand_augment = keras_cv.layers.RandAugment(
    value_range=(0, 255),
    augmentations_per_image=3,
    magnitude=0.8,
)


def load_and_preprocess_image(path, label, image_size, augment=False):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, image_size)
    if augment:
        image = tf.cast(image, tf.uint8)
        image = rand_augment(image)
        image = tf.cast(image, tf.float32) / 255.0
        # image = data_augmentation(image)  # Optionally add your other augmentations here
    else:
        image = tf.cast(image, tf.float32) / 255.0
    return image


def get_tf_dataset(
    files,
    regression=False,
    class_to_idx=None,
    min_year=None,
    max_year=None,
    augment=False,
    image_size=(224, 224),
    shuffle=True,
    include_paths=False,
):
    image_paths, labels = create_dataset(files)
    image_paths = tf.constant(image_paths)

    if regression:
        labels = [(y - min_year) / (max_year - min_year) for y in labels]
        labels = tf.constant(labels, dtype=tf.float32)
    else:
        labels = [class_to_idx[y] for y in labels]
        labels = tf.constant(labels, dtype=tf.int32)
        # One-hot
        # labels = tf.one_hot(indices=labels_idx, depth=len(class_to_idx))

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


def get_input_shape(model_name):
    if model_name == "NASNetMobile":
        return (224, 224, 3)
    elif model_name == "ResNet101":
        # return (224, 224, 3)
        return (384, 384, 3)
    elif model_name == "InceptionV3":
        return (299, 299, 3)
        # return (384, 384, 3)
    elif model_name == "EfficientNetB3":
        return (255, 255, 3)
    elif model_name == "ConvNeXtTiny":
        return (224, 224, 3)
    elif model_name == "EfficientNetV2S":
        return (384, 384, 3)
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


@register_keras_serializable()
def ordinal_categorical_cross_entropy(y_true, y_pred):
    num_classes = tf.cast(tf.shape(y_pred)[-1], tf.float32)
    true_labels = tf.argmax(y_true, axis=-1)
    pred_labels = tf.argmax(y_pred, axis=-1)
    weights = tf.abs(tf.cast(pred_labels - true_labels, tf.float32)) / (num_classes - 1.0)

    # Apply label smoothing
    # smooth_y_true = y_true * (1.0 - 0.1) + 0.1 / num_classes

    # base_loss = tf.keras.losses.sparse_categorical_crossentropy(smooth_y_true, y_pred)
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

    val_ds_unshuffled = get_tf_dataset(
        [test_file],
        regression=regression,
        class_to_idx=class_to_idx,
        min_year=min_year,
        max_year=max_year,
        augment=False,
        image_size=input_shape[:2],
        shuffle=False,
        include_paths=True,
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
        val_ds_unshuffled_batched = val_ds_unshuffled.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

        # Build and compile model
        model = build_model(
            config,
            num_classes=num_classes,
            input_shape=input_shape,
        )

        optimizer = tf.keras.optimizers.Adam(learning_rate=config["training"]["learning_rate"])
        loss = "mean_squared_error" if regression else ordinal_categorical_cross_entropy
        # Don't forget to use one-hot men using focal
        # loss = tf.keras.losses.CategoricalFocalCrossentropy(
        #     alpha=0.25,  # Can be a scalar or a list with per-class weights
        #     gamma=2.0,  # Controls focus on hard examples (higher = more focus)
        # )
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
            preds = model.predict(val_ds_unshuffled_batched, verbose=0)
            preds_years = preds * (max_year - min_year) + min_year
            preds_years_rounded = np.round(preds_years).astype(int)

            # Collect true labels from the batched dataset
            y_true = []
            image_paths = []
            for _, labels_batch, paths_batch in val_ds_unshuffled_batched:
                y_true.extend(labels_batch.numpy())
                image_paths.extend([p.numpy().decode("utf-8") for p in paths_batch])

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

            # Perform misclassification analysis
            misclass_analysis = analyze_misclassifications(y_true_years_rounded, preds_years_rounded, image_paths)

            # Print analysis results
            print(f"\n{fold_print}Misclassification Analysis:")
            print(
                f"Near misses (within 2 years): {misclass_analysis['num_near_misses']} out of {misclass_analysis['num_misclassified']} misclassifications ({misclass_analysis['percent_near_misses']:.2f}%)"
            )
            print(f"MAE with outliers: {misclass_analysis['original_mae']:.2f}")
            print(
                f"MAE without outliers: {misclass_analysis['non_outlier_mae']:.2f} (improvement: {misclass_analysis['mae_improvement']:.2f})"
            )
            print(f"\n10 Worst misclassifications:")
            for img_path, true_year, pred_year, error in misclass_analysis["worst_misclassifications"]:
                print(f"Image: {img_path}, True: {true_year}, Predicted: {pred_year}, Error: {error}")

    else:
        BATCH_SIZE = config["training"]["batch_size"]
        train_ds_batched = train_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
        val_ds_batched = val_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
        val_ds_unshuffled_batched = val_ds_unshuffled.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

        # Build and compile model
        model = build_model(
            config,
            num_classes=num_classes,
            input_shape=input_shape,
        )

        optimizer = tf.keras.optimizers.Adam(learning_rate=config["training"]["learning_rate"])
        loss = "mean_squared_error" if regression else ordinal_categorical_cross_entropy
        # loss = tf.keras.losses.CategoricalFocalCrossentropy(
        #     alpha=0.25,  # Can be a scalar or a list with per-class weights
        #     gamma=2.0,  # Controls focus on hard examples (higher = more focus)
        # )
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
            preds = model.predict(val_ds_unshuffled_batched, verbose=0)
            preds_years = preds * (max_year - min_year) + min_year
            preds_years_rounded = np.round(preds_years).astype(int)

            # Collect true labels from the batched dataset
            y_true = []
            image_paths = []
            for _, labels_batch, paths_batch in val_ds_unshuffled_batched:
                y_true.extend(labels_batch.numpy())
                image_paths.extend([p.numpy().decode("utf-8") for p in paths_batch])

            exact_matches = np.sum(preds_years_rounded.flatten() == y_true_years_rounded.flatten())
            total = len(y_true_years_rounded)
            mae = np.mean(np.abs(preds_years_rounded.flatten() - y_true_years_rounded.flatten()))
            print(f"{fold_print}Exactly correct year predictions: {exact_matches} out of {total}")
            print(f"{fold_print}Final MAE (rounded to years): {mae:.2f}")
            metrics = {"mae": mae, "exact": exact_matches, "total": total}

            os.makedirs("plots", exist_ok=True)

            plot_metrics(history, fold_str, plot_dir, regression, class_to_idx)

            # Perform misclassification analysis
            misclass_analysis = analyze_misclassifications(y_true_years_rounded, preds_years_rounded, image_paths)

            # Print analysis results
            print(f"\n{fold_print}Misclassification Analysis:")
            print(
                f"Near misses (within 2 years): {misclass_analysis['num_near_misses']} out of {misclass_analysis['num_misclassified']} misclassifications ({misclass_analysis['percent_near_misses']:.2f}%)"
            )
            print(f"MAE with outliers: {misclass_analysis['original_mae']:.2f}")
            print(
                f"MAE without outliers: {misclass_analysis['non_outlier_mae']:.2f} (improvement: {misclass_analysis['mae_improvement']:.2f})"
            )
            print(f"\n5 Worst misclassifications:")
            for img_path, true_year, pred_year, error in misclass_analysis["worst_misclassifications"]:
                print(f"Image: {img_path}, True: {true_year}, Predicted: {pred_year}, Error: {error}")

        else:
            # Collect predictions
            image_paths = []
            y_score = []
            y_true = []
            y_pred = []
            for images, labels, paths in val_ds_unshuffled_batched:
                preds = model.predict(images, verbose=0)
                y_true.extend(labels.numpy())
                y_pred.extend(np.argmax(preds, axis=1))
                y_score.extend(preds)
                image_paths.extend([p.numpy().decode("utf-8") for p in paths])
            y_score = np.array(y_score)

            mcc = matthews_corrcoef(y_true, y_pred)

            plot_metrics(history, fold_str, plot_dir, regression, class_to_idx, y_true, y_pred, y_score)

            if class_to_idx is not None:
                idx_to_class = {v: k for k, v in class_to_idx.items()}
                y_true_years = np.array([idx_to_class[idx] for idx in y_true])
                y_pred_years = np.array([idx_to_class[idx] for idx in y_pred])
                mae_years = np.mean(np.abs(y_true_years - y_pred_years))
                print(f"Classification MAE (in years): {mae_years:.2f}")

                # Perform misclassification analysis
                misclass_analysis = analyze_misclassifications(y_true_years, y_pred_years, image_paths)

                # Print analysis results
                print(f"\n{fold_print}Misclassification Analysis:")
                print(
                    f"Near misses (within 2 years): {misclass_analysis['num_near_misses']} out of {misclass_analysis['num_misclassified']} misclassifications ({misclass_analysis['percent_near_misses']:.2f}%)"
                )
                print(f"MAE with outliers: {misclass_analysis['original_mae']:.2f}")
                print(
                    f"MAE without outliers: {misclass_analysis['non_outlier_mae']:.2f} (improvement: {misclass_analysis['mae_improvement']:.2f})"
                )
                print(f"\n5 Worst misclassifications:")
                for img_path, true_year, pred_year, error in misclass_analysis["worst_misclassifications"]:
                    print(f"Image: {img_path}, True: {true_year}, Predicted: {pred_year}, Error: {error}")

            metrics = {"accuracy": results[1], "mae_years": mae_years, "mcc": mcc}
    return metrics


def analyze_misclassifications(y_true, y_pred, image_paths, threshold=2, num_worst=10):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    image_paths = np.array(image_paths)

    errors = np.abs(y_true - y_pred)

    # Get misclassifications and near misses
    misclassified_idx = np.where(errors > 0)[0]
    near_misses_idx = np.where((errors > 0) & (errors <= threshold))[0]

    total_samples = len(y_true)
    num_misclassified = len(misclassified_idx)
    num_near_misses = len(near_misses_idx)

    # Get the worst misclassifications
    worst_idx = np.argsort(errors)[-num_worst:][::-1]
    worst_errors = [(image_paths[i], int(y_true[i]), int(y_pred[i]), int(errors[i])) for i in worst_idx]

    # Calculate MAE with and without outliers
    original_mae = np.mean(errors)

    # Define outliers as samples with errors greater than mean + 2*std
    mean_error = np.mean(errors)
    std_error = np.std(errors)
    outlier_threshold = mean_error + 2 * std_error
    outliers_idx = np.where(errors > outlier_threshold)[0]

    # Calculate MAE without outliers
    if len(outliers_idx) > 0:
        non_outlier_mae = np.mean(errors[errors <= outlier_threshold])
    else:
        non_outlier_mae = original_mae

    return {
        "total_samples": total_samples,
        "num_misclassified": num_misclassified,
        "num_near_misses": num_near_misses,
        "percent_near_misses": 100 * num_near_misses / num_misclassified if num_misclassified > 0 else 0,
        "worst_misclassifications": worst_errors,
        "original_mae": original_mae,
        "outlier_threshold": outlier_threshold,
        "num_outliers": len(outliers_idx),
        "non_outlier_mae": non_outlier_mae,
        "mae_improvement": original_mae - non_outlier_mae,
        "outliers": [(image_paths[i], int(y_true[i]), int(y_pred[i]), int(errors[i])) for i in outliers_idx],
    }


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
        x = tf.keras.layers.Dense(128, activation="relu")(x)
        x = tf.keras.layers.BatchNormalization()(x)
    elif model_name == "ConvNeXtTiny":
        base_model = tf.keras.applications.ConvNeXtTiny(
            weights="imagenet", input_shape=input_shape, include_top=False, include_preprocessing=True
        )
        base_model.trainable = False
        x = base_model.output
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(128, activation="relu")(x)
        x = tf.keras.layers.BatchNormalization()(x)
    elif model_name == "EfficientNetV2S":
        base_model = tf.keras.applications.EfficientNetV2S(
            weights="imagenet",
            input_shape=input_shape,
            include_top=False,
            include_preprocessing=True,  # Let TF handle the preprocessing
        )
        base_model.trainable = False
        x = base_model.output
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(128, activation="relu")(x)
        x = tf.keras.layers.BatchNormalization()(x)
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
    regression = config.get("task", "classification") == "regression"
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

    elif config["null_hypothesis_testing"]["use"]:
        # if config["null_hypothesis_testing"]:
        #     raise ValueError("config['null_hypothesis_testing'] and config['cross_validation'] cannot both be set to True")

        from scipy import stats
        import copy

        print("\n===== Running 5x2 Cross-Validation for Null Hypothesis Testing =====")
        
        # Load all data
        all_files = [f"data/datasets/fold{fold}.csv" for fold in fold_nums]
        all_df = pd.concat([pd.read_csv(file) for file in all_files])
        
        # Choose different models for comparison

        # Create deep copies to ensure no shared references
        base_config = copy.deepcopy(config)
        alt_config = copy.deepcopy(config)

        # Get model names directly from config file
        base_model_name = config["null_hypothesis_testing"]["model_1"]
        alt_model_name = config["null_hypothesis_testing"]["model_2"]

        # Set the specified models in the configs
        base_config["model"]["name"] = base_model_name
        alt_config["model"]["name"] = alt_model_name

        print(f"Using models specified in config: {base_model_name} vs {alt_model_name}")
        print(f"Comparing: Base model ({base_config['model']['name']}) vs. "
            f"Alternative model ({alt_config['model']['name']})")
        
        # Run 5 iterations of 2-fold cross-validation
        base_metrics = []
        alt_metrics = []
        difference_metrics = []
        
        for iteration in range(5):
            print(f"\n===== Iteration {iteration+1}/5 =====")
            
            # Shuffle and split data into two folds
            all_df_shuffled = all_df.sample(frac=1, random_state=iteration).reset_index(drop=True)
            split_idx = len(all_df_shuffled) // 2
            fold1 = all_df_shuffled.iloc[:split_idx]
            fold2 = all_df_shuffled.iloc[split_idx:]
            
            # Save temporary fold files
            temp_fold1_path = "data/datasets/temp_fold1.csv"
            temp_fold2_path = "data/datasets/temp_fold2.csv" 
            fold1.to_csv(temp_fold1_path, index=False)
            fold2.to_csv(temp_fold2_path, index=False)

            # Get unique years and setup for classification
            all_years = all_df_shuffled["year"].unique().tolist()
            classes = sorted(list(set(all_years)))
            class_to_idx = {y: i for i, y in enumerate(classes)}
            num_classes = len(classes)
            min_year = None
            max_year = None
            
            # Run 2-fold CV with base model
            print("=== Training Base Model ===")
            # Train on fold1, test on fold2
            metrics_base_1 = train_and_evaluate(
                [temp_fold1_path], temp_fold2_path, 
                class_to_idx, num_classes, min_year, max_year,
                base_config, f"{run_id}_base_{base_config['model']['name']}_iter{iteration}_fold1", f"{iteration}_1"
            )
            
            # Train on fold2, test on fold1
            metrics_base_2 = train_and_evaluate(
                [temp_fold2_path], temp_fold1_path,
                class_to_idx, num_classes, min_year, max_year,
                base_config, f"{run_id}_base_{base_config['model']['name']}_iter{iteration}_fold2", f"{iteration}_2"
            )
            
            # Run 2-fold CV with alternative model
            print("=== Training Alternative Model ===")
            # Train on fold1, test on fold2
            metrics_alt_1 = train_and_evaluate(
                [temp_fold1_path], temp_fold2_path,
                class_to_idx, num_classes, min_year, max_year,
                alt_config, f"{run_id}_alt_{alt_config['model']['name']}_iter{iteration}_fold1", f"{iteration}_1"
            )
            
            # Train on fold2, test on fold1
            metrics_alt_2 = train_and_evaluate(
                [temp_fold2_path], temp_fold1_path,
                class_to_idx, num_classes, min_year, max_year,
                alt_config, f"{run_id}_alt_{alt_config['model']['name']}_iter{iteration}_fold2", f"{iteration}_2"
            )
            
            # Calculate and store differences in performance
            base_metrics.extend([metrics_base_1["accuracy"], metrics_base_2["accuracy"]])
            alt_metrics.extend([metrics_alt_1["accuracy"], metrics_alt_2["accuracy"]])
            
            diff1 = metrics_base_1["accuracy"] - metrics_alt_1["accuracy"]
            diff2 = metrics_base_2["accuracy"] - metrics_alt_2["accuracy"]
            difference_metrics.append((diff1, diff2))
            
            # Clean up temp files
            os.remove(temp_fold1_path)
            os.remove(temp_fold2_path)
        
        # Calculate 5x2cv paired t-test statistic
        mean_diff = np.mean([d[0] for d in difference_metrics])
        variance_sum = np.sum([np.var([d[0], d[1]]) for d in difference_metrics])
        t_statistic = mean_diff / np.sqrt(variance_sum / 5)
        
        # Calculate p-value (approximate using t-distribution with 5 degrees of freedom)
        p_value = 2 * (1 - stats.t.cdf(abs(t_statistic), df=5))

        standard_t, standard_p = stats.ttest_rel(base_metrics, alt_metrics)
        
        print("\n===== 5x2 Cross-Validation Results =====")
        print(f"Base model average accuracy: {np.mean(base_metrics):.4f}")
        print(f"Alternative model average accuracy: {np.mean(alt_metrics):.4f}")
        print(f"Mean difference: {mean_diff:.4f}")

        # Print results for specialized 5x2cv t-test
        print("\nSpecialized 5x2cv t-test (Dietterich 1998):")
        print(f"t-statistic: {t_statistic:.4f}")
        print(f"p-value: {p_value:.4f}")
        print(f"Statistically significant difference: {'Yes' if p_value < 0.05 else 'No'} (alpha=0.05)")
        
        # Print results for standard paired t-test
        print("\nStandard paired t-test:")
        print(f"t-statistic: {standard_t:.4f}")
        print(f"p-value: {standard_p:.4f}")
        print(f"Statistically significant difference: {'Yes' if standard_p < 0.05 else 'No'} (alpha=0.05)")

        # Save results
        results_dir = os.path.join("results", "5x2cv")
        os.makedirs(results_dir, exist_ok=True)
        with open(os.path.join(results_dir, f"{run_id}_{base_config['model']['name']}-{alt_config['model']['name']}_5x2cv_results.txt"), "w") as f:
            f.write("5x2 Cross-Validation Results\n")
            f.write("===========================\n")
            f.write(f"Base model: {base_config['model']['name']} with {base_config['model']['dense_units']} units\n")
            f.write(f"Alt model: {alt_config['model']['name']} with {alt_config['model']['dense_units']} units\n\n")
            f.write(f"Base model average accuracy: {np.mean(base_metrics):.4f}\n")
            f.write(f"Alternative model average accuracy: {np.mean(alt_metrics):.4f}\n")
            f.write(f"Mean difference: {mean_diff:.4f}\n")

            # Write specialized 5x2cv t-test results
            f.write("Specialized 5x2cv t-test ( Dietterich 1998):\n")
            f.write(f"t-statistic: {t_statistic:.4f}\n")
            f.write(f"p-value: {p_value:.4f}\n")
            f.write(f"Statistically significant difference: {'Yes' if p_value < 0.05 else 'No'} (alpha=0.05)\n\n")
            
            # Write standard t-test results
            f.write("Standard paired t-test:\n")
            f.write(f"t-statistic: {standard_t:.4f}\n")
            f.write(f"p-value: {standard_p:.4f}\n")
            f.write(f"Statistically significant difference: {'Yes' if standard_p < 0.05 else 'No'} (alpha=0.05)\n") 

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
