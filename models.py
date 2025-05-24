import tensorflow as tf
from keras.saving import register_keras_serializable
import numpy as np


def get_input_shape(model_name):
    if model_name == "NASNetMobile":
        return (224, 224, 3)  # max size
    elif model_name == "ResNet101":
        return (224, 224, 3)  # optional, can be larger
    elif model_name == "InceptionV3":
        return (224, 224, 3)
        # return (299, 299, 3)  # optional, can be larger
        # return (384, 384, 3)
        # return (448, 448, 3)
    elif model_name == "ConvNeXtTiny":
        # return (224, 224, 3)
        # return (384, 384, 3)
        return (448, 448, 3)
    elif model_name == "EfficientNetV2S":
        return (384, 384, 3)
    else:
        raise ValueError(f"Unsupported model name: {model_name}")


@register_keras_serializable()
def ordinal_categorical_cross_entropy(y_true, y_pred):
    """Custom loss function for classification that penalizes larger classification errors more."""
    num_classes = tf.cast(tf.shape(y_pred)[-1], tf.float32)
    true_labels = tf.argmax(y_true, axis=-1)
    pred_labels = tf.argmax(y_pred, axis=-1)
    weights = tf.abs(tf.cast(pred_labels - true_labels, tf.float32)) / (num_classes - 1.0)

    base_loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
    loss = (1.0 + weights) * base_loss
    return loss


@register_keras_serializable()
def ordinal_regression_loss(y_true, y_pred):
    """Custom loss function for regression that penalizes larger year differences more"""
    base_loss = tf.abs(y_true - y_pred)  # MAE base
    # Apply increasing penalty for larger errors
    scaled_loss = base_loss * (1.0 + base_loss)
    return tf.reduce_mean(scaled_loss)


def create_year_mae_loss(min_year, max_year):
    """Create a loss function that computes MAE in year units."""
    year_range = max_year - min_year

    def year_mae_loss(y_true, y_pred, config=None):
        norm_method = config.get("model", {}).get("normalization_method", "minmax") if config else "minmax"

        if norm_method == "standardize":
            # For standardized data, need labels_mean and labels_std from training set
            labels_mean = tf.constant(np.mean([min_year, max_year]), dtype=tf.float32)
            labels_std = tf.constant(np.std(np.arange(min_year, max_year + 1)), dtype=tf.float32)

            y_true_years = y_true * labels_std + labels_mean
            y_pred_years = y_pred * labels_std + labels_mean
        else:
            # Default to min-max scaling
            y_true_years = y_true * year_range + min_year
            y_pred_years = y_pred * year_range + min_year

        return tf.reduce_mean(tf.abs(y_true_years - y_pred_years))

    return year_mae_loss


def setup_model(config, num_classes, input_shape, regression=False):
    """Build and compile model with appropriate loss and metrics"""
    model = build_model(config, num_classes=num_classes, input_shape=input_shape)

    # Different optimizer for regression and classification
    if regression and config.get("training", {}).get("use_adamw", False):
        # Use AdamW for regression if configured
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=config["training"]["learning_rate"],
            weight_decay=config.get("training", {}).get("weight_decay", 1e-4),
        )
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate=config["training"]["learning_rate"])

    loss = ordinal_regression_loss if regression else ordinal_categorical_cross_entropy
    metrics_list = ["mae", "mse"] if regression else ["accuracy"]

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics_list)
    return model


def build_model(config, num_classes=None, input_shape=(224, 224, 3)):
    """Build model architecture based on configuration."""
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
    elif model_name == "ResNet101":
        base_model = tf.keras.applications.ResNet101(weights="imagenet", input_shape=input_shape, include_top=False)
        base_model.trainable = False
        x = base_model.output
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
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
        x = tf.keras.layers.BatchNormalization()(x)
    elif model_name == "ConvNeXtTiny":
        base_model = tf.keras.applications.ConvNeXtTiny(
            weights="imagenet", input_shape=input_shape, include_top=False, include_preprocessing=True
        )
        base_model.trainable = False
        x = base_model.output
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.BatchNormalization()(x)
    elif model_name == "EfficientNetV2S":
        base_model = tf.keras.applications.EfficientNetV2S(
            weights="imagenet",
            input_shape=input_shape,
            include_top=False,
            include_preprocessing=True,
        )
        base_model.trainable = False
        x = base_model.output
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.BatchNormalization()(x)
    else:
        raise ValueError(f"Unsupported model name: {model_name}")

    if regression:
        x = tf.keras.layers.Dense(128, activation="relu")(x)
        if config["model"].get("dropout", {}).get("use", False):
            x = tf.keras.layers.Dropout(config["model"].get("dropout", {}).get("value", 0))(x)
        x = tf.keras.layers.BatchNormalization()(x)

        x = tf.keras.layers.Dense(64, activation="relu")(x)
        if config["model"].get("dropout", {}).get("use", False):
            x = tf.keras.layers.Dropout(config["model"].get("dropout", {}).get("value", 0))(x)
        x = tf.keras.layers.BatchNormalization()(x)

        if config["model"].get("l2_regularization", {}).get("use", False):
            predictions = tf.keras.layers.Dense(
                1,
                activation="linear",
                name="PREDICTIONS",
                kernel_regularizer=tf.keras.regularizers.l2(
                    float(config["model"].get("l2_regularization", {}).get("value", 0))
                ),
            )(x)
        else:
            predictions = tf.keras.layers.Dense(1, activation="linear", name="PREDICTIONS")(x)
    else:
        # Add dense layers based on config
        for _ in range(dense_layers):
            x = tf.keras.layers.Dense(dense_units, activation="relu")(x)
            # Optional dropout after each dense layer
            if config["model"].get("dropout", {}).get("use", False):
                x = tf.keras.layers.Dropout(config["model"].get("dropout", {}).get("value", 0))(x)

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
