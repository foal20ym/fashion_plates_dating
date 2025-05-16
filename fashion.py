import tensorflow as tf
import pandas as pd
import numpy as np
import time
import os

# Import from our modules
from models import build_model
from data import get_tf_dataset
from training import train_and_evaluate
from utils import load_config, setup_gpu, format_time
from plotting import plot_cv_metrics_summary


def prepare_data_for_fold(train_folds, test_fold, fold_nums, regression=False):
    """Prepare data files and class mapping for a specific fold."""
    train_files = [f"data/datasets/fold{fold}.csv" for fold in train_folds]
    test_file = f"data/datasets/fold{test_fold}.csv"

    if not regression:
        # Classification: Create class mapping
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
        # Regression: Find min and max years
        all_years = []
        for file in train_files + [test_file]:
            df = pd.read_csv(file)
            all_years.extend(df["year"].tolist())
        min_year = min(all_years)
        max_year = max(all_years)
        class_to_idx = None
        num_classes = None

    return train_files, test_file, class_to_idx, num_classes, min_year, max_year


def run_cross_validation(config, run_id, fold_nums):
    """Run 10-fold cross-validation."""
    regression = config.get("task", "classification") == "regression"
    model_name = config.get("model", {}).get("name", "InceptionV3")

    all_metrics = []
    for test_fold in fold_nums:
        print(f"\n===== Fold {test_fold} =====")
        train_folds = [fold for fold in fold_nums if fold != test_fold]

        # Prepare data for this fold
        train_files, test_file, class_to_idx, num_classes, min_year, max_year = prepare_data_for_fold(
            train_folds, test_fold, fold_nums, regression
        )

        # Train and evaluate on this fold
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

    # Generate summary plots
    plot_cv_metrics_summary(
        all_metrics, plot_dir=os.path.join("plots", "10_fold_cv", f"{run_id}_{model_name}_mean_std_var")
    )

    # Print aggregate metrics
    print_aggregate_metrics(all_metrics, regression)

    return all_metrics


def run_single_fold(config, run_id, fold_nums):
    """Run training on a single fold."""
    regression = config.get("task", "classification") == "regression"

    # Use the first fold as test fold
    test_fold = fold_nums[0]
    print(f"Test fold: {test_fold}")
    train_folds = [fold for fold in fold_nums if fold != test_fold]

    # Prepare data
    train_files, test_file, class_to_idx, num_classes, min_year, max_year = prepare_data_for_fold(
        train_folds, test_fold, fold_nums, regression
    )

    # Train and evaluate
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

    return metrics


def print_aggregate_metrics(all_metrics, regression):
    """Print summary metrics from cross-validation."""
    if regression:
        maes = [m["mae"] for m in all_metrics]
        exacts = [m["exact"] for m in all_metrics]
        totals = [m["total"] for m in all_metrics]
        exact_percent = 100 * np.mean(exacts) / np.mean(totals)

        print(f"\nMean MAE over all folds: {np.mean(maes):.2f} ± {np.std(maes):.2f}")
        print(f"Mean exact matches: {np.mean(exacts):.2f} / {np.mean(totals):.2f} ({exact_percent:.2f}%)")
    else:
        accs = [m["accuracy"] for m in all_metrics]
        maes = [m["mae_years"] for m in all_metrics]
        mccs = [m["mcc"] for m in all_metrics]

        print(f"\nMean MAE over all folds: {np.mean(maes):.4f} ± {np.std(maes):.4f}")
        print(f"Mean accuracy over all folds: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
        print(f"Mean Matthews Correlation Coefficient: {np.mean(mccs):.4f} ± {np.std(mccs):.4f}")


def main():
    """Main function to run the fashion plate dating model."""
    # Set up GPU
    setup_gpu()

    start_time = time.time()

    # Load configuration
    config = load_config("config.yaml")
    regression = config.get("task", "classification") == "regression"
    model_name = config.get("model", {}).get("name", "InceptionV3")

    # Initialize fold numbers and run ID
    fold_nums = list(range(10))  # 10 folds (0-9)
    run_id = time.strftime("%Y-%m-%d_%H:%M:%S")

    print(f"\n=== Using model: {model_name}. ===")
    print(f"RUN ID: {run_id}")
    print(f"Task: {'Regression' if regression else 'Classification'}")

    # Run cross-validation or single fold based on config
    if config["cross_validation"]:
        all_metrics = run_cross_validation(config, run_id, fold_nums)
    else:
        metrics = run_single_fold(config, run_id, fold_nums)

    # Print execution time
    end_time = time.time()
    running_time = end_time - start_time
    hours, minutes, seconds = format_time(running_time)
    print(f"\n=== Total running time: {hours} hours, {minutes} minutes, {seconds} seconds ===\n")


if __name__ == "__main__":
    main()
