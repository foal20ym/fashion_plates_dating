import yaml
import os
import numpy as np
from sklearn.utils.class_weight import compute_class_weight


def load_config(config_path="../config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_highest_version_for_saved_model(model_name):
    """Get the next version number for saving a model."""
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


def get_class_weights(labels, method="balanced", max_weight=0.75):
    """Calculate class weights for imbalanced datasets."""
    classes = np.array(sorted(set(labels)))
    weights = compute_class_weight(class_weight=method, classes=classes, y=labels)

    capped_weights = {cls: min(w, max_weight) for cls, w in zip(classes, weights)}
    return capped_weights


def setup_gpu():
    """Configure GPU settings for optimal performance."""
    import tensorflow as tf

    print("TensorFlow Version:", tf.__version__)
    print("Num GPUs Available:", len(tf.config.list_physical_devices("GPU")))
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            # Use only the first GPU
            tf.config.set_visible_devices(gpus[0], "GPU")
            # Allow memory growth to prevent hogging all GPU memory
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("Using GPU:", gpus[0])
        except RuntimeError as e:
            print(e)
    else:
        print("No GPU detected. Training will run on CPU.")


def format_time(seconds):
    """Format time in seconds to hours, minutes, seconds format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return hours, minutes, seconds
