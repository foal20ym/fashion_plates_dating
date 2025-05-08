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
