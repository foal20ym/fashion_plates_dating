import tensorflow as tf
import os
import pandas as pd
import numpy as np
from sklearn.metrics import matthews_corrcoef

# Import from other modules
from data import get_tf_dataset, prepare_datasets
from models import setup_model
from metrics import (
    collect_regression_predictions,
    collect_classification_predictions,
    analyze_misclassifications,
    print_misclassification_analysis,
    calculate_metrics,
)
from utils import get_class_weights, get_highest_version_for_saved_model
from plotting import plot_metrics, plot_class_distribution
from tuning import run_hyperparameter_tuning, apply_best_hyperparameters


def create_callbacks(config, log_dir):
    """Set up training callbacks"""
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=config["training"]["early_stopping_patience"],
            restore_best_weights=True,
            min_delta=1e-4,
            verbose=1,
        ),
        tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=config["training"]["reduce_lr_patience"],
            min_delta=1e-4,
            verbose=1,
            min_lr=1e-6,
        ),
    ]

    # Add cosine decay learning rate schedule for regression if configured
    if config.get("task", "classification") == "regression" and config.get("training", {}).get("cosine_decay", False):
        callbacks.append(
            tf.keras.callbacks.LearningRateScheduler(
                lambda epoch: config["training"]["learning_rate"]
                * (0.1 + 0.9 * tf.math.cos(epoch / config["training"]["epochs"] * 3.14159))
            )
        )

    return callbacks


def prepare_directories(config, run_id, model_name, fold_idx):
    """Set up directories for logs and plots."""
    fold_str = f"_fold{fold_idx}" if fold_idx is not None else ""

    if config.get("cross_validation", False) and fold_idx is not None:
        plot_dir = os.path.join("plots", "10_fold_cv", f"{run_id}_{model_name}_fold{fold_idx}")
        log_dir = os.path.join("logs", "tensorboard", "10_fold_cv", f"{run_id}_fit_fold{fold_idx}")
    else:
        plot_dir = os.path.join("plots", "single_run", f"{run_id}_{model_name}_fold{fold_idx}")
        log_dir = os.path.join("logs", "tensorboard", f"fit_{run_id}_{model_name}_fit_fold{fold_idx}")

    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs("plots", exist_ok=True)

    return plot_dir, log_dir, fold_str


def train_and_evaluate(train_files, test_file, class_to_idx, num_classes, min_year, max_year, config, run_id, fold_idx):
    """Train and evaluate a model with given configuration."""
    idx_to_class = {v: k for k, v in class_to_idx.items()} if class_to_idx is not None else None
    input_shape = get_input_shape(config.get("model", {}).get("name", "InceptionV3"))
    model_name = config.get("model", {}).get("name", "InceptionV3")
    regression = config.get("task", "classification") == "regression"

    # Helper for fold-specific naming
    fold_print = f"Fold {fold_idx} " if fold_idx is not None else ""

    # Set up directories
    plot_dir, log_dir, fold_str = prepare_directories(config, run_id, model_name, fold_idx)

    # Prepare datasets
    train_ds = get_tf_dataset(
        train_files,
        config,
        regression=regression,
        class_to_idx=class_to_idx,
        min_year=min_year,
        max_year=max_year,
        augment=True,
        image_size=input_shape[:2],
    )
    val_ds = get_tf_dataset(
        [test_file],
        config,
        regression=regression,
        class_to_idx=class_to_idx,
        min_year=min_year,
        max_year=max_year,
        augment=False,
        image_size=input_shape[:2],
    )

    val_ds_unshuffled = get_tf_dataset(
        [test_file],
        config,
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
        # Extract labels
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
        config = apply_best_hyperparameters(config, best_params)
        print(
            f"{fold_print}Using tuned parameters: batch_size={config['training']['batch_size']}, "
            + f"dropout={config['model']['dropout']}, lr={config['training']['learning_rate']}"
        )

    # Prepare dataset with batching
    BATCH_SIZE = config["training"]["batch_size"]
    train_ds_batched, val_ds_batched, val_ds_unshuffled_batched = prepare_datasets(
        train_ds, val_ds, val_ds_unshuffled, BATCH_SIZE
    )

    # Build and compile model
    model = setup_model(config, num_classes, input_shape, regression)

    # Create callbacks
    callbacks = create_callbacks(config, log_dir)

    # Save model if needed
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

    metrics = {}

    # Process predictions and calculate metrics
    if regression:
        # Handle regression task
        y_true_years_rounded, preds_years_rounded, image_paths = collect_regression_predictions(
            model, val_ds_unshuffled_batched, min_year, max_year
        )

        # Calculate metrics
        exact_matches = np.sum(preds_years_rounded.flatten() == y_true_years_rounded.flatten())
        total = len(y_true_years_rounded)
        mae = np.mean(np.abs(preds_years_rounded.flatten() - y_true_years_rounded.flatten()))

        print(f"{fold_print}Exactly correct year predictions: {exact_matches} out of {total}")
        print(f"{fold_print}Final MAE (rounded to years): {mae:.2f}")

        metrics = {"mae": mae, "exact": exact_matches, "total": total}

        # Plot metrics
        plot_metrics(history, fold_str, plot_dir, regression, class_to_idx)

        # Analyze misclassifications
        misclass_analysis = analyze_misclassifications(y_true_years_rounded, preds_years_rounded, image_paths)
        print_misclassification_analysis(misclass_analysis, fold_print)

    else:
        # Handle classification task
        y_true, y_pred, y_score, image_paths = collect_classification_predictions(model, val_ds_unshuffled_batched)
        mcc = matthews_corrcoef(y_true, y_pred)

        # Plot metrics
        plot_metrics(history, fold_str, plot_dir, regression, class_to_idx, y_true, y_pred, y_score)

        if class_to_idx is not None:
            idx_to_class = {v: k for k, v in class_to_idx.items()}
            y_true_years = np.array([idx_to_class[idx] for idx in y_true])
            y_pred_years = np.array([idx_to_class[idx] for idx in y_pred])
            mae_years = np.mean(np.abs(y_true_years - y_pred_years))

            print(f"Classification MAE (in years): {mae_years:.2f}")

            # Analyze misclassifications
            misclass_analysis = analyze_misclassifications(y_true_years, y_pred_years, image_paths)
            print_misclassification_analysis(misclass_analysis, fold_print)

            metrics = {"accuracy": results[1], "mae_years": mae_years, "mcc": mcc}

    return metrics
