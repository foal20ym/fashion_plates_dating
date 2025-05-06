def plot_results(
    history,
    results,
    y_score,
    y_true,
    y_pred,
    config,
    plot_dir,
    class_to_idx,
    fold_str,
    regression,
    min_year=None,
    max_year=None,
):
    import matplotlib.pyplot as plt
    import seaborn as sns

    os.makedirs(plot_dir, exist_ok=True)
    if regression:
        preds_years = y_score * (max_year - min_year) + min_year
        y_true_years = y_true * (max_year - min_year) + min_year
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
        # Add more regression plots as needed
    else:
        target_names = [str(k) for k in sorted(class_to_idx.keys(), key=lambda x: class_to_idx[x])]
        all_labels = [class_to_idx[k] for k in sorted(class_to_idx.keys(), key=lambda x: class_to_idx[x])]
        cm = confusion_matrix(y_true, y_pred, labels=all_labels)
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=False, fmt="d", cmap="Blues", xticklabels=target_names, yticklabels=target_names)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(f"Confusion Matrix{fold_str}")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"confusion_matrix.png"))
        plt.close()
        # Add more classification plots as needed
