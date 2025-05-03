import math
import os

import cv2
import numpy as np
import pandas as pd
import pdb
import tensorflow as tf


class FoldGenerator(tf.keras.utils.Sequence):

    '''Data generator for training and testing based on given fold CSV files'''

    YEAR_COLUMN = 'year'
    FILE_COLUMN = 'file'

    MIN_YEAR = 1820
    MAX_YEAR = 1880

    @classmethod
    def __split_train_valid(cls, base_data, seed, valid_ratio):
        '''Split loaded data into training and validation set

        Parameters
        ----------
        base_data : Entire training set
        seed : Seed value to use when splitting into train and validation
        valid_ratio : Ratio of data to use for validation

        Returns
        -------
        train_data : Selected training data
        valid_data : Selected validation data

        '''
        train_data = None
        valid_data = None

        base_data = base_data.sample(frac=1, random_state=seed)
        counts = base_data[cls.YEAR_COLUMN].value_counts()
        for year, _ in counts.sort_values().items():
            rows = base_data[base_data[cls.YEAR_COLUMN] == year]
            valid_count = max(1, math.floor(len(rows) * valid_ratio))

            if valid_data is None:
                valid_data = rows.iloc[:valid_count, :]
                train_data = rows.iloc[valid_count:, :]
            else:
                valid_data = valid_data.append(rows.iloc[:valid_count, :])
                train_data = train_data.append(rows.iloc[valid_count:, :])

        return train_data, valid_data

    @classmethod
    def create_training_generators(cls, fold_dir, test_fold, seed,
                                   valid_ratio=0.1):
        '''Create training and validation generator based on given fold CSV
        files

        Parameters
        ----------
        fold_dir : Folder containing fold CSV files
        test_fold : Fold number used for testing
        seed : Seed value to use when splitting into train and validation
        valid_ratio : Ratio of data to use for validation

        Returns
        -------
        train_generator : Generator for training
        valid_generator : Generator for validation

        '''
        csv_files = [os.path.join(fold_dir, f) for f in os.listdir(fold_dir)
                     if f.endswith('csv') and str(test_fold) not in f]

        base_data = None

        for file in csv_files:
            if base_data is None:
                base_data = pd.read_csv(file)
            else:
                base_data = base_data.append(pd.read_csv(file),
                                             ignore_index=True)

        train_data, valid_data = cls.__split_train_valid(base_data, seed,
                                                         valid_ratio)

        # add data folder location to image paths
        train_data[cls.FILE_COLUMN] = train_data[cls.FILE_COLUMN].apply(
                                                    lambda x: os.path.join(
                                                                fold_dir, x))
        valid_data[cls.FILE_COLUMN] = valid_data[cls.FILE_COLUMN].apply(
                                                    lambda x: os.path.join(
                                                                fold_dir, x))
        train_gen = FoldGenerator(train_data, True, seed)
        valid_gen = FoldGenerator(valid_data, False, seed)

        return train_gen, valid_gen

    @classmethod
    def create_test_generator(cls, fold_dir, test_fold, batch_size=1):
        '''Create test generator based on given fold CSV files

        Parameters
        ----------
        fold_dir : Folder containing fold CSV files
        test_fold : Fold number used for testing
        batch_size : Number of images per batch

        Returns
        -------
        test_generator : Generator for testing

        '''
        csv_file = [os.path.join(fold_dir, f) for f in os.listdir(fold_dir)
                    if f.endswith('csv') and str(test_fold) in f][0]
        data = pd.read_csv(csv_file)

        # add data folder location to image paths
        data[cls.FILE_COLUMN] = data[cls.FILE_COLUMN].apply(lambda x:
                                                            os.path.join(
                                                                fold_dir, x))

        return FoldGenerator(data, False, 42, batch_size,
                             return_file_path=True)

    @classmethod
    def unnormalize_year(cls, year):
        '''Convert predicted value into year number

        Parameters
        ----------
        year : Predicted year

        Returns
        -------
        Year number

        '''
        return year + cls.MIN_YEAR

    def __init__(self, data, augment, seed, batch_size=16, img_size=(224, 224),
                 return_file_path=False):
        '''Create data generator

        Parameters
        ----------
        data : DataFrame of images to use in format [(img_path, year)]
        augment : Apply augmentation to returned images
        seed : Seed value to use
        batch_size : Number of images per batch
        img_size : Size of the images to be returned
        return_file_path : Let generator also return image path for
                           identification

        '''
        tf.keras.utils.Sequence.__init__(self)

        self._data = data
        self._augment = augment
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self._batch_size = batch_size
        self._img_size = img_size
        self._return_file_path = return_file_path
        self.on_epoch_end()

    def __len__(self):
        '''Number of batches per epoch
        Returns
        -------
        Number of batches per epoch

        '''
        return len(self._data) // self._batch_size

    def __getitem__(self, index):
        '''Generate one batch of images

        Parameters
        ----------
        index : Batch id

        Returns
        -------
        Inputs for this batch, expected targets

        '''
        indexes = self._indexes[index*self._batch_size:
                                (index+1)*self._batch_size]
        selected = self._data.iloc[indexes, :]

        X, y, paths = self.__generate_data(selected)

        if self._return_file_path:
            return X, y, paths

        return X, y

    def on_epoch_end(self):
        '''Update indexes after each epoch
        '''
        self._indexes = np.arange(len(self._data))
        self._rng.shuffle(self._indexes)

    def __generate_data(self, selected):
        '''Prepare batch from selected image paths

        Parameters
        ----------
        selected : Selected image paths

        Returns
        -------
        Inputs for this batch, expected targets, image paths

        '''
        X = []
        y = []
        paths = []

        for _, row in selected.iterrows():
            path = row[self.FILE_COLUMN]
            year = row[self.YEAR_COLUMN]

            X.append(self.__load_img(path))
            y.append(self.__normalize_year(year))
            paths.append(path)

        return np.array(X), np.array(y), paths

    def __normalize_year(self, year):
        '''Normalize year number

        Parameters
        ----------
        year : Unnormalized year

        Returns
        -------
        Normalized year number

        '''
        return year - self.MIN_YEAR

    def __load_img(self, img_path):
        '''Load image from disk

        Parameters
        ----------
        img_path : Path to the image to load

        Returns
        -------
        Loaded (and augmented) image

        '''
        img = cv2.imread(img_path)
        img = img / 255.
        img = cv2.resize(img, self._img_size)

        if self._augment:
            select_augment = self._rng.integers(0, 1, size=3, endpoint=True)
            if select_augment[0] == 1:
                select_flip = self._rng.integers(0, 1, endpoint=True)
                img = cv2.flip(img, select_flip)

            if select_augment[1] == 1:
                angle = self._rng.integers(0, 144, endpoint=True) - 72
                center = np.array(self._img_size) / 2
                transform_matrix = cv2.getRotationMatrix2D(tuple(center),
                                                           angle, 1)
                border_value = 0
                img = cv2.warpAffine(img, transform_matrix,
                                     dsize=self._img_size,
                                     borderValue=border_value)

            if select_augment[2] == 1:
                factor = self._rng.uniform(0.85, 1.15)
                mean = img.mean(axis=(0, 1))
                img = (img - mean) * factor + mean

        return img

    def image_size(self):
        '''Get the size of the images returned by the generator
        Returns
        -------
        Size of the images

        '''
        return self._img_size
