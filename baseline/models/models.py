import os

import pandas as pd
import tensorflow as tf


class Model():

    '''Basic model class'''

    def __init__(self, input_size, model_path=None):
        '''Setup model

        Parameters
        ----------
        input_size : Size of the input images
        model_path : Path to saved model, optional


        '''
        self._input_size = input_size
        self._base_model = None
        if model_path is None:
            self._model = self._setup_model()
        else:
            self._model = tf.keras.models.load_model(model_path)

    def _setup_model(self):
        '''Setup model
           NOTE: to be implemented by each subclass
        Returns
        -------
        Compiled model

        '''
        raise NotImplementedError()

    def train(self, train_generator, valid_generator, log_dir, epochs=100):
        '''Train model

        Parameters
        ----------
        train_generator : Generator for training images
        valid_generator : Generator for validation images
        log_dir : Folder to which models and training progress is logged to
        epochs : Number of epochs to train for at most, optional

        '''
        # train with frozen weights
        callbacks = [tf.keras.callbacks.CSVLogger(
                                            os.path.join(log_dir,
                                                         'frozen_log.csv'))]
        self._model.fit(train_generator, epochs=10,
                        validation_data=valid_generator, callbacks=callbacks)

        # fine-tune model
        self._base_model.trainable = True
        callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience=10),
                     tf.keras.callbacks.ModelCheckpoint(
                                            os.path.join(log_dir,
                                                         'trained.h5')),
                     tf.keras.callbacks.CSVLogger(
                                            os.path.join(log_dir,
                                                         'fine_log.csv'))]

        self._model.compile(optimizer='adam', loss='mse',
                            metrics=[tf.keras.metrics.mean_absolute_error])
        self._model.fit(train_generator, epochs=epochs,
                        validation_data=valid_generator, callbacks=callbacks)

    def test(self, test_generator):
        '''Test model

        Parameters
        ----------
        test_generator : Generator for test images

        Returns
        -------
        DataFrame with [file, predicted, gt]

        '''
        results = {'file': [], 'predicted': [], 'gt': []}

        iterator = iter(test_generator)

        for img, gt, path in iterator:
            predicted = self._model.predict(img)
            results['file'] += path

            real_predicted = [test_generator.unnormalize_year(f)
                              for f in predicted]
            results['predicted'] += real_predicted

            real_gt = [test_generator.unnormalize_year(f)
                       for f in gt]
            results['gt'] += real_gt

        return pd.DataFrame(results)


class NASNetModel(Model):

    '''NASNet Mobile Model with one layer and unfrozen, pre-trained weights'''

    def _setup_model(self):
        '''Setup the NASNet model
        Returns
        -------
        Compiled model

        '''
        self._base_model = tf.keras.applications.NASNetMobile(
                                                weights='imagenet',
                                                input_shape=(*self._input_size,
                                                             3),
                                                include_top=False)
        self._base_model.trainable = False

        inputs = tf.keras.Input(shape=(*self._input_size, 3))

        x = self._base_model(inputs, training=False)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        # Dense layer size found through Bayesian optimization
        x = tf.keras.layers.Dense(76, activation='relu')(x)
        outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

        model = tf.keras.Model(inputs, outputs)
        model.compile(optimizer='adam', loss='mse',
                      metrics=[tf.keras.metrics.mean_absolute_error])

        return model


class ResNet101Model(Model):

    '''ResNet101 Model with one layer and unfrozen, pre-trained weights'''

    def _setup_model(self):
        '''Setup the ResNet101 model
        Returns
        -------
        Compiled model

        '''
        self._base_model = tf.keras.applications.ResNet101(
                                                weights='imagenet',
                                                input_shape=(*self._input_size,
                                                             3),
                                                include_top=False)
        self._base_model.trainable = False

        inputs = tf.keras.Input(shape=(*self._input_size, 3))

        x = self._base_model(inputs, training=False)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        # Dense layer size found through Bayesian optimization
        x = tf.keras.layers.Dense(10, activation='relu')(x)
        outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

        model = tf.keras.Model(inputs, outputs)
        model.compile(optimizer='adam', loss='mse',
                      metrics=[tf.keras.metrics.mean_absolute_error])

        return model


class InceptionModel(Model):

    '''Inception Model with no additional layer and frozen,
       pre-trained weights'''

    def _setup_model(self):
        '''Setup the Inception model
        Returns
        -------
        Compiled model

        '''
        self._base_model = tf.keras.applications.InceptionV3(
                                                weights='imagenet',
                                                input_shape=(*self._input_size,
                                                             3),
                                                include_top=False)
        self._base_model.trainable = False

        inputs = tf.keras.Input(shape=(*self._input_size, 3))

        x = self._base_model(inputs, training=False)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

        model = tf.keras.Model(inputs, outputs)
        model.compile(optimizer='adam', loss='mse',
                      metrics=[tf.keras.metrics.mean_absolute_error])

        return model
