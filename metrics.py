import numpy as np
import tensorflow as tf
from sklearn.metrics import matthews_corrcoef


def collect_regression_predictions(model, val_ds_unshuffled_batched, min_year, max_year):
    """Collect predictions and true labels for regression tasks."""
    # Collect true labels and image paths from the batched dataset
    y_true = []
    image_paths = []
    model_inputs = []

    # Extract data from the dataset
    for images, labels, paths in val_ds_unshuffled_batched:
        model_inputs.append(images)
        y_true.extend(labels.numpy())
        image_paths.extend([p.numpy().decode("utf-8") for p in paths])

    # Predict on batches separately and concatenate results
    preds = []
    for inputs in model_inputs:
        batch_preds = model.predict(inputs, verbose=0)
        preds.append(batch_preds)

    preds = np.vstack(preds)

    # Convert predictions to year values
    preds_years = preds * (max_year - min_year) + min_year
    preds_years_rounded = np.round(preds_years).astype(int)

    # Convert labels to year values
    y_true = np.array(y_true)
    y_true_years = y_true * (max_year - min_year) + min_year
    y_true_years_rounded = np.round(y_true_years).astype(int)

    return y_true_years_rounded, preds_years_rounded, image_paths


def collect_classification_predictions(model, val_ds_unshuffled_batched):
    """Collect predictions and true labels for classification tasks"""
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

    return np.array(y_true), np.array(y_pred), np.array(y_score), image_paths


def analyze_misclassifications(y_true, y_pred, image_paths, threshold=2, num_worst=10):
    """Analyze prediction errors and misclassifications."""
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    image_paths = np.array(image_paths)

    errors = np.abs(y_true - y_pred)

    # Get misclassifications and near misses
    misclassified_idx = np.where(errors > 0)[0]
    near_misses_idx = np.where((errors > 0) & (errors <= threshold))[0]
    big_misses_idx = np.where(errors >= (threshold * 5))[0]

    total_samples = len(y_true)
    num_misclassified = len(misclassified_idx)
    num_near_misses = len(near_misses_idx)
    num_big_misses = len(big_misses_idx)

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
        "num_big_misses": num_big_misses,
        "percent_near_misses": 100 * num_near_misses / num_misclassified if num_misclassified > 0 else 0,
        "worst_misclassifications": worst_errors,
        "original_mae": original_mae,
        "outlier_threshold": outlier_threshold,
        "num_outliers": len(outliers_idx),
        "non_outlier_mae": non_outlier_mae,
        "mae_improvement": original_mae - non_outlier_mae,
        "outliers": [(image_paths[i], int(y_true[i]), int(y_pred[i]), int(errors[i])) for i in outliers_idx],
    }


def print_misclassification_analysis(misclass_analysis, fold_print=""):
    """Print formatted misclassification analysis results."""
    print(f"\n{fold_print}Misclassification Analysis:")
    print(
        f"Near misses (within 2 years): {misclass_analysis['num_near_misses']} out of {misclass_analysis['num_misclassified']} misclassifications ({misclass_analysis['percent_near_misses']:.2f}%)"
    )
    print(f"Big misses (greater than 10 years): {misclass_analysis['num_big_misses']}")
    print(f"MAE with outliers: {misclass_analysis['original_mae']:.2f}")
    print(
        f"MAE without outliers: {misclass_analysis['non_outlier_mae']:.2f} (improvement: {misclass_analysis['mae_improvement']:.2f})"
    )
    print(f"\n10 Worst misclassifications:")
    for img_path, true_year, pred_year, error in misclass_analysis["worst_misclassifications"]:
        print(f"Image: {img_path}, True: {true_year}, Predicted: {pred_year}, Error: {error}")


def calculate_metrics(results, y_true, y_pred, y_true_years=None, y_pred_years=None):
    """Calculate metrics for the model performance."""
    metrics = {}

    if y_true_years is not None and y_pred_years is not None:
        # For both regression and classification with year conversion
        mae_years = np.mean(np.abs(y_true_years - y_pred_years))
        exact_matches = np.sum(y_pred_years == y_true_years)
        total = len(y_true_years)

        metrics = {"mae": mae_years, "exact": exact_matches, "total": total}

        if len(results) > 1:  # Classification metrics
            metrics["accuracy"] = results[1]
            metrics["mcc"] = matthews_corrcoef(y_true, y_pred)
            metrics["mae_years"] = mae_years

    return metrics
