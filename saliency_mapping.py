from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
from tf_keras_vis.utils.scores import CategoricalScore
from keras.saving import register_keras_serializable
from tf_keras_vis.saliency import Saliency
from keras.models import load_model
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
import time
import yaml
import os


@register_keras_serializable()
def ordinal_categorical_cross_entropy(y_true, y_pred):
    num_classes = tf.cast(tf.shape(y_pred)[-1], tf.float32)
    true_labels = tf.argmax(y_true, axis=-1)
    pred_labels = tf.argmax(y_pred, axis=-1)
    weights = tf.abs(tf.cast(pred_labels - true_labels, tf.float32)) / (num_classes - 1.0)
    base_loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
    loss = (1.0 + weights) * base_loss
    return loss
    

def show_sailency_map():
    
    saved_model = load_model(
        "trained_models/InceptionV3_version_1.keras",
        custom_objects={"ordinal_categorical_cross_entropy": ordinal_categorical_cross_entropy}
    )
    saved_model.summary()

    img_path = 'data/datasets/private/1860/1860_79et.jpg'
    input_shape = saved_model.input_shape[1:3]
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, input_shape)
    img = tf.cast(img, tf.float32) / 255.0
    img_array = img.numpy()
    img_array = np.expand_dims(img_array, axis=0)

    # Define score function for the class index you want to visualize
    # Here, we use the top predicted class
    preds = saved_model.predict(img_array)
    class_idx = np.argmax(preds[0])
    score = CategoricalScore(class_idx)

    saliency = Saliency(saved_model,
                        model_modifier=ReplaceToLinear(),
                        clone=True)

    # Compute saliency map
    saliency_map = saliency(score, img_array)
    sal = saliency_map[0]
    sal = (sal - sal.min()) / (sal.max() - sal.min() + 1e-8)  # Normalize to [0, 1]
    sal = np.power(sal, 0.4)

    # Detta får ut bild-id't 
    img_filename = os.path.basename(img_path)
    img_id = os.path.splitext(img_filename)[0]

    model_dir = "saliency_maps"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        

    # Plot original image and saliency map
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow((img.numpy() * 255).astype(np.uint8))
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    #axes[1].imshow(saliency_map[0], cmap='jet')
    axes[1].imshow(sal, cmap='jet')
    axes[1].set_title("Saliency Map")
    axes[1].axis('off')
    plt.tight_layout()
    plt.savefig(f"saliency_maps/saliency_map_{img_id}.png")
    plt.show()


show_sailency_map()