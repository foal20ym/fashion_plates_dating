import os
import pandas as pd
import numpy as np
from scipy import stats
import copy

from training import train_and_evaluate


def run_5x2_cross_validation(config, run_id, fold_nums, train_and_evaluate_fn=None):
    """Run 5x2 cross-validation for statistical model comparison (null hypothesis testing)."""

    # if config["null_hypothesis_testing"]:
    #     raise ValueError("config['null_hypothesis_testing'] and config['cross_validation'] cannot both be set to True")

    if train_and_evaluate_fn is None:
        train_and_evaluate_fn = train_and_evaluate

    print("\n===== Running 5x2 Cross-Validation for Model Comparison =====")

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
    print(
        f"Comparing: Base model ({base_config['model']['name']}) vs. "
        f"Alternative model ({alt_config['model']['name']})"
    )

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
            [temp_fold1_path],
            temp_fold2_path,
            class_to_idx,
            num_classes,
            min_year,
            max_year,
            base_config,
            f"{run_id}_base_{base_config['model']['name']}_iter{iteration}_fold1",
            f"{iteration}_1",
        )

        # Train on fold2, test on fold1
        metrics_base_2 = train_and_evaluate(
            [temp_fold2_path],
            temp_fold1_path,
            class_to_idx,
            num_classes,
            min_year,
            max_year,
            base_config,
            f"{run_id}_base_{base_config['model']['name']}_iter{iteration}_fold2",
            f"{iteration}_2",
        )

        # Run 2-fold CV with alternative model
        print("=== Training Alternative Model ===")
        # Train on fold1, test on fold2
        metrics_alt_1 = train_and_evaluate(
            [temp_fold1_path],
            temp_fold2_path,
            class_to_idx,
            num_classes,
            min_year,
            max_year,
            alt_config,
            f"{run_id}_alt_{alt_config['model']['name']}_iter{iteration}_fold1",
            f"{iteration}_1",
        )

        # Train on fold2, test on fold1
        metrics_alt_2 = train_and_evaluate(
            [temp_fold2_path],
            temp_fold1_path,
            class_to_idx,
            num_classes,
            min_year,
            max_year,
            alt_config,
            f"{run_id}_alt_{alt_config['model']['name']}_iter{iteration}_fold2",
            f"{iteration}_2",
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
    with open(
        os.path.join(
            results_dir, f"{run_id}_{base_config['model']['name']}-{alt_config['model']['name']}_5x2cv_results.txt"
        ),
        "w",
    ) as f:
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

    return {
        "base_metrics": base_metrics,
        "alt_metrics": alt_metrics,
        "mean_diff": mean_diff,
        "t_statistic": t_statistic,
        "p_value": p_value,
        "standard_t": standard_t,
        "standard_p": standard_p,
    }
