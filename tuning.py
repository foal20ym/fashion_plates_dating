import os
import yaml
import tensorflow as tf
from kerastuner import BayesianOptimization, RandomSearch, Hyperband
from kerastuner.engine.hyperparameters import HyperParameters
from models import ordinal_categorical_cross_entropy, ordinal_regression_loss, get_input_shape


def create_hypermodel(hp, config, num_classes=None, input_shape=(224, 224, 3)):
    regression = config["task"] == "regression"
    model_name = config["model"]["name"]
    tuning_config = config["hyperparameter_tuning"]

    # 1. Dropout rate
    dropout_rate = hp.Float(
        "dropout",
        min_value=tuning_config["parameters"]["dropout"]["min"],
        max_value=tuning_config["parameters"]["dropout"]["max"],
        step=tuning_config["parameters"]["dropout"]["step"],
    )

    # 2. Learning rate
    learning_rate = hp.Float(
        "learning_rate",
        min_value=tuning_config["parameters"]["learning_rate"]["min"],
        max_value=tuning_config["parameters"]["learning_rate"]["max"],
        sampling=tuning_config["parameters"]["learning_rate"]["sampling"],
    )

    # 3. L2 regularization strength (if enabled)
    if tuning_config["parameters"].get("l2_regularization", {}).get("tune", False):
        l2_values = [float(val) for val in tuning_config["parameters"]["l2_regularization"]["values"]]
        l2_reg = hp.Choice("l2_regularization", l2_values)
    else:
        l2_reg = float(config["model"].get("l2_regularization", 0.0))

    # 4. Fine tuning depth (number of layers to unfreeze from the end)
    if tuning_config["parameters"].get("fine_tune_layers", {}).get("tune", False):
        fine_tune_layers = hp.Int(
            "fine_tune_layers",
            min_value=tuning_config["parameters"]["fine_tune_layers"]["min"],
            max_value=tuning_config["parameters"]["fine_tune_layers"]["max"],
            step=tuning_config["parameters"]["fine_tune_layers"]["step"],
        )
    else:
        fine_tune_layers = 1

    # 5. Dense layers configuration
    if tuning_config["parameters"].get("dense_units", {}).get("tune", False):
        dense_units = hp.Choice("dense_units", tuning_config["parameters"]["dense_units"]["values"])
    else:
        dense_units = tuning_config["parameters"].get("dense_units", {}).get("default", 128)

    if tuning_config["parameters"].get("dense_layers", {}).get("tune", False):
        dense_layers = hp.Int(
            "dense_layers",
            min_value=tuning_config["parameters"]["dense_layers"]["min"],
            max_value=tuning_config["parameters"]["dense_layers"]["max"],
            step=tuning_config["parameters"]["dense_layers"]["step"],
        )
    else:
        dense_layers = tuning_config["parameters"].get("dense_layers", {}).get("default", 1)

    # Select base model and setup according to model_name
    if model_name == "NASNetMobile":
        base_model = tf.keras.applications.NASNetMobile(weights="imagenet", input_shape=input_shape, include_top=False)
    elif model_name == "ResNet101":
        base_model = tf.keras.applications.ResNet101(weights="imagenet", input_shape=input_shape, include_top=False)
    elif model_name == "InceptionV3":
        base_model = tf.keras.applications.InceptionV3(weights="imagenet", input_shape=input_shape, include_top=False)
    elif model_name == "EfficientNetB3":
        base_model = tf.keras.applications.EfficientNetB3(
            weights="imagenet", input_shape=input_shape, include_top=False
        )
    elif model_name == "ConvNeXtTiny":
        base_model = tf.keras.applications.ConvNeXtTiny(
            weights="imagenet", input_shape=input_shape, include_top=False, include_preprocessing=True
        )
    elif model_name == "EfficientNetV2S":  # Add this block
        base_model = tf.keras.applications.EfficientNetV2S(
            weights="imagenet",
            input_shape=input_shape,
            include_top=False,
            include_preprocessing=True,
        )
    else:
        raise ValueError(f"Unsupported model name: {model_name}")

    # Freeze base model layers
    base_model.trainable = False
    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    # Add dense layers based on hyperparameters
    for _ in range(dense_layers):
        x = tf.keras.layers.Dense(dense_units, activation="relu")(x)
        if dropout_rate > 0:
            x = tf.keras.layers.Dropout(dropout_rate)(x)

    # Output layer
    if regression:
        if l2_reg > 0:
            predictions = tf.keras.layers.Dense(
                1,
                activation="linear",  # Changed from sigmoid to linear
                name="PREDICTIONS",
                kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
            )(x)
        else:
            predictions = tf.keras.layers.Dense(1, activation="linear", name="PREDICTIONS")(x)
    else:
        if l2_reg > 0:
            predictions = tf.keras.layers.Dense(
                num_classes,
                activation="softmax",
                name="PREDICTIONS",
                kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
            )(x)
        else:
            predictions = tf.keras.layers.Dense(num_classes, activation="softmax", name="PREDICTIONS")(x)

    model = tf.keras.models.Model(inputs=base_model.input, outputs=predictions)

    # Fine-tuning: unfreeze specified number of layers
    if fine_tune_layers > 0:
        for layer in base_model.layers[-fine_tune_layers:]:
            layer.trainable = True

    # Compile with tunable learning rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss = ordinal_regression_loss if regression else ordinal_categorical_cross_entropy
    metrics = ["mae", "mse"] if regression else ["accuracy"]
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    return model


def run_hyperparameter_tuning(train_ds, val_ds, config, num_classes, run_id, fold_idx=None):
    fold_str = f"_fold{fold_idx}" if fold_idx is not None else ""
    model_name = config["model"]["name"]
    input_shape = get_input_shape(model_name)

    tuning_config = config["hyperparameter_tuning"]
    method = tuning_config["method"]
    max_trials = tuning_config["max_trials"]
    executions_per_trial = tuning_config["executions_per_trial"]
    directory = f"{tuning_config['directory']}_{run_id}{fold_str}"
    project_name = f"{tuning_config['project_name']}_{model_name}{fold_str}"

    # Early stopping callback for each trial
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=config["training"]["early_stopping_patience"] // 2,  # Shorter patience for tuning
            restore_best_weights=True,
            min_delta=1e-4,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=config["training"]["reduce_lr_patience"] // 2,  # Shorter patience for tuning
            min_delta=1e-4,
            verbose=1,
            min_lr=1e-6,
        ),
    ]

    # Create hypermodel wrapper function
    def build_hypermodel(hp):
        return create_hypermodel(hp, config, num_classes, input_shape)

    # Choose tuner based on method
    if method == "random":
        print("Using Random Search for hyperparameter tuning")
        tuner = RandomSearch(
            build_hypermodel,
            objective="val_loss",
            max_trials=max_trials,
            executions_per_trial=executions_per_trial,
            directory=directory,
            project_name=project_name,
        )
    elif method == "bayesian":
        print("Using Bayesian Optimization for hyperparameter tuning")
        tuner = BayesianOptimization(
            build_hypermodel,
            objective="val_loss",
            max_trials=max_trials,
            executions_per_trial=executions_per_trial,
            directory=directory,
            project_name=project_name,
        )
    elif method == "hyperband":
        print("Using Hyperband for hyperparameter tuning")
        tuner = Hyperband(
            build_hypermodel,
            objective="val_loss",
            max_epochs=config["training"]["epochs"],
            factor=3,
            directory=directory,
            project_name=project_name,
        )
    else:
        raise ValueError(f"Unsupported tuning method: {method}")

    # Get all batch sizes to test
    batch_sizes = tuning_config["parameters"]["batch_size"]["values"]

    best_params = {}
    best_val_loss = float("inf")
    best_batch_size = None

    print(f"\n===== Starting Hyperparameter Tuning with {method} search =====")
    print(f"Max trials: {max_trials}, Executions per trial: {executions_per_trial}")

    # Loop through batch sizes (batch size needs special handling as it affects dataset creation)
    for batch_size in batch_sizes:
        print(f"\nTuning with batch size: {batch_size}")

        # Create batched datasets with current batch size
        train_ds_batched = train_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        val_ds_batched = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

        # Set this batch size for all models in this tuning round
        hp = HyperParameters()
        hp.Fixed("batch_size", batch_size)
        tuner.search_space = hp

        # Run tuning for current batch size
        tuner.search(
            train_ds_batched,
            validation_data=val_ds_batched,
            epochs=config["training"]["epochs"],
            callbacks=callbacks,
            verbose=2,
        )

        # Check if this batch size gave better results
        best_trial = tuner.oracle.get_best_trials(1)[0]
        trial_val_loss = best_trial.score

        if trial_val_loss < best_val_loss:
            best_val_loss = trial_val_loss
            best_batch_size = batch_size
            best_hps = tuner.get_best_hyperparameters(1)[0]

    # Collect all best hyperparameters
    best_params = {
        "batch_size": best_batch_size,
        "learning_rate": best_hps.get("learning_rate"),
        "dropout": best_hps.get("dropout"),
        "val_loss": best_val_loss,
    }

    # Add optional hyperparameters if they were tuned
    if "l2_regularization" in best_hps.values:
        best_params["l2_regularization"] = best_hps.get("l2_regularization")

    if "fine_tune_layers" in best_hps.values:
        best_params["fine_tune_layers"] = best_hps.get("fine_tune_layers")

    if "dense_units" in best_hps.values:
        best_params["dense_units"] = best_hps.get("dense_units")

    if "dense_layers" in best_hps.values:
        best_params["dense_layers"] = best_hps.get("dense_layers")

    # Print best hyperparameters
    print("\n===== Best Hyperparameters =====")
    for param, value in best_params.items():
        print(f"{param}: {value}")

    # Save results to file
    os.makedirs("hyperparameter_tuning_results", exist_ok=True)
    with open(f"hyperparameter_tuning_results/best_params_{run_id}{fold_str}.yaml", "w") as f:
        yaml.dump(best_params, f)

    return best_params


def apply_best_hyperparameters(config, best_params):
    modified_config = config.copy()

    # Update config with best parameters
    modified_config["training"]["batch_size"] = best_params["batch_size"]
    modified_config["training"]["learning_rate"] = best_params["learning_rate"]
    modified_config["model"]["dropout"] = best_params["dropout"]

    # Apply optional hyperparameters if they were tuned
    if "l2_regularization" in best_params:
        modified_config["model"]["l2_regularization"] = best_params["l2_regularization"]
        modified_config["model"]["use_l2_regularization"] = True

    if "fine_tune_layers" in best_params:
        modified_config["model"]["fine_tune"] = best_params["fine_tune_layers"] > 0
        modified_config["model"]["fine_tune_layers"] = best_params["fine_tune_layers"]

    if "dense_units" in best_params:
        modified_config["model"]["dense_units"] = best_params["dense_units"]

    if "dense_layers" in best_params:
        modified_config["model"]["dense_layers"] = best_params["dense_layers"]

    return modified_config
