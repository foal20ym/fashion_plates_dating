from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, RandomRotation, RandomFlip, RandomContrast
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import InceptionV3, ResNet101, NASNetMobile
from tensorflow.keras.regularizers import l2
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
from plotting import plot_cv_metrics_summary, plot_metrics
import keras_cv
#VAR TVUNGEN ATT KÖRA: pip install --upgrade keras-cv-nightly tf-nightly


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
        #image = data_augmentation(image)  # Optionally add your other augmentations here
    else:
        image = tf.cast(image, tf.float32) / 255.0
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


def ordinal_categorical_cross_entropy(y_true, y_pred):
    num_classes = tf.cast(tf.shape(y_pred)[-1], tf.float32)
    true_labels = tf.argmax(y_true, axis=-1)
    pred_labels = tf.argmax(y_pred, axis=-1)
    weights = tf.abs(tf.cast(pred_labels - true_labels, tf.float32)) / (num_classes - 1.0)
    base_loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
    loss = (1.0 + weights) * base_loss
    return loss
    

def train_and_evaluate(train_files, test_file, class_to_idx, num_classes, min_year, max_year, config, run_id, fold_idx):
    input_shape = get_input_shape(config["model"]["name"])
    model_name = config["model"]["name"]
    regression = config["task"] == "regression"

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
    model.compile(optimizer=optimizer, loss=ordinal_categorical_cross_entropy, metrics=metrics)

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

        plot_metrics(history, fold_str, plot_dir, regression, class_to_idx, y_true, y_pred, y_score)

        if class_to_idx is not None:
            idx_to_class = {v: k for k, v in class_to_idx.items()}
            y_true_years = np.array([idx_to_class[idx] for idx in y_true])
            y_pred_years = np.array([idx_to_class[idx] for idx in y_pred])
            mae_years = np.mean(np.abs(y_true_years - y_pred_years))
            print(f"Classification MAE (in years): {mae_years:.2f}")

        metrics = {"accuracy": results[1], "mae_years": mae_years}
    return metrics


def build_model(config, num_classes=None, input_shape=(224, 224, 3)):
    regression = config["task"] == "regression"
    model_name = config["model"]["name"]
    fine_tune = config["model"]["fine_tune"]

    # Select base model and setup according to model_name
    if model_name == "NASNetMobile":
        base_model = NASNetMobile(weights="imagenet", input_shape=input_shape, include_top=False)
        base_model.trainable = False
        x = base_model.output
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(76, activation="relu")(x)
    elif model_name == "ResNet101":
        base_model = ResNet101(weights="imagenet", input_shape=input_shape, include_top=False)
        base_model.trainable = False
        x = base_model.output
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(10, activation="relu")(x)
    elif model_name == "InceptionV3":
        base_model = InceptionV3(weights="imagenet", input_shape=input_shape, include_top=False)
        base_model.trainable = False
        x = base_model.output
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
    else:
        raise ValueError(f"Unsupported model name: {model_name}")

    # Optional dropout
    if config["model"]["use_dropout"]:
        x = Dropout(config["model"]["dropout"])(x)

    # Output layer
    if regression:
        if config["model"]["use_l2_regularization"]:
            predictions = Dense(
                1,
                activation="sigmoid",
                name="PREDICTIONS",
                kernel_regularizer=l2(float(config["model"]["l2_regularization"])),
            )(x)
        else:
            predictions = Dense(1, activation="sigmoid", name="PREDICTIONS")(x)
    else:
        if config["model"]["use_l2_regularization"]:
            predictions = Dense(
                num_classes,
                activation="softmax",
                name="PREDICTIONS",
                kernel_regularizer=l2(float(config["model"]["l2_regularization"])),
            )(x)
        else:
            predictions = Dense(num_classes, activation="softmax", name="PREDICTIONS")(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    # Fine-tuning: unfreeze last layer if requested
    if fine_tune:
        print("Fine Tuning!")
        for layer in base_model.layers[-1:]:
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
    regression = config["task"] == "regression"
    model_name = config["model"]["name"]
    fold_nums = [num for num in range(10)]
    run_id = time.strftime("%Y-%m-%d_%H:%M:%S")
    print(f"Using model: {model_name}.")
    print(f"RUN ID: {run_id}")

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
            print(f"\nMean accuracy over 10 folds: {np.mean(accs):.4f}")

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
    print(f"Total running time: {hours} hours, {minutes} minutes, {seconds} seconds")


if __name__ == "__main__":
    main()
