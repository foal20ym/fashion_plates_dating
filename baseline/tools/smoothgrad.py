import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def generate_noisy_images(image, num_samples, noise_level):
    images = image.reshape((1, *image.shape))
    images = np.repeat(images, num_samples, axis=0)
    noise = noise_level * (np.max(image) - np.min(image))
    noise_images = np.random.normal(0, noise, images.shape).astype(np.float32)

    return images + noise_images


def output_gradients(noisy_images, model):
    with tf.GradientTape() as tape:
        inputs = tf.cast(noisy_images, tf.float32)
        tape.watch(inputs)
        predictions = model(inputs)

    return tape.gradient(predictions, inputs)


def normalize_gradients(gradients):
    np_mean = np.mean(gradients, axis=0)
    np_unscaled = np.sum(np.abs(np_mean), axis=-1)

    return np_unscaled / np.max(np_unscaled)


def main(model_path, image_path, out_path, samples, noise_levels):

    model = tf.keras.models.load_model(model_path)

    image = cv2.imread(image_path)
    image = image / 255.
    image = cv2.resize(image, (224, 224))

    out_folder = os.path.join(out_path,
                              os.path.basename(image_path).split('.')[0])
    if not os.path.isdir(out_folder):
        os.mkdir(out_folder)

    cv2.imwrite(os.path.join(out_folder, 'original.png'), image*255)

    noise_levels = [float(f) for f in noise_levels.split(',')]

    for noise_level in noise_levels:
        noisy_images = generate_noisy_images(image, samples, noise_level)
        gradients = output_gradients(noisy_images, model)
        normalized = normalize_gradients(gradients)

        plt.imsave(os.path.join(out_folder, f'smoothgrad_n{noise_level}.png'),
                   normalized, cmap='Reds')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
                      description='Visualize pixel relevance using SmoothGrad',
                      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('model_path', help='Path to trained model')
    parser.add_argument('image_path', help='Path to image to explain')
    parser.add_argument('out_path', help='Path to root folder')
    parser.add_argument('--samples', help='Number of samples to generate',
                        default=64, type=int)
    parser.add_argument('--noise_levels', help='Percentage of noise to add',
                        default='0.1,0.2')

    args = vars(parser.parse_args())
    main(**args)
