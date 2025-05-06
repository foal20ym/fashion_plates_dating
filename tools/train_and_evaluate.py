def train_and_evaluate(train_files, test_file, class_to_idx, num_classes, min_year, max_year, config, fold_idx=None):
    input_shape = get_input_shape(config["model"]["name"], config["model"]["include_top"])
    regression = config["task"] == "regression"
    fold_str = f"_fold{fold_idx}" if fold_idx is not None else ""
    run_id = time.strftime("%Y-%m-%d_%H:%M:%S")
    if config.get("cross_validation", False) and fold_idx is not None:
        plot_dir = os.path.join("plots", "10_fold_cv", f"{run_id}_{config['model']['name']}_fold{fold_idx}")
        log_dir = os.path.join("logs", "tensorboard", "10_fold_cv", f"{run_id}_fit_fold{fold_idx}")
    else:
        plot_dir = "plots"
        log_dir = os.path.join("logs", "tensorboard", f"fit_{run_id}")
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    config["log_dir"] = log_dir

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

    model, history = train_model(train_ds_batched, val_ds_batched, config, num_classes, input_shape)
    if regression:
        results, preds, y_true = evaluate_model(model, val_ds_batched, config, class_to_idx, min_year, max_year)
        plot_results(
            history,
            results,
            preds,
            y_true,
            None,
            config,
            plot_dir,
            class_to_idx,
            fold_str,
            regression,
            min_year,
            max_year,
        )
        return {"mae": results[0]}
    else:
        results, y_score, y_true, y_pred = evaluate_model(
            model, val_ds_batched, config, class_to_idx, min_year, max_year
        )
        plot_results(history, results, y_score, y_true, y_pred, config, plot_dir, class_to_idx, fold_str, regression)
        return {"accuracy": results[1]}


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
