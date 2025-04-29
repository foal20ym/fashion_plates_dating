from keras.applications import NASNetLarge
from keras.optimizers import SGD, Adam
import random
import pandas as pd
import numpy as np
import tensorflow as tf

""" 
Här först så väljer jag bara ut 9st av de 10 tillängliga folds 
genom att välja 9st random nummer.
"""
fold_nums = [num for num in range(10)]
print(fold_nums)

test_fold = random.choice(fold_nums)
print(f"Test fold: {test_fold}")

train_folds = [fold for fold in fold_nums if fold != test_fold]
print(f"Train folds: {train_folds}")

train_files = [f"../datasets/fold{fold}.csv" for fold in train_folds]
test_file = f"../datasets/fold{test_fold}.csv"

fold_files = train_files
test_files = [test_file]

image_size = (299, 299)
#batch_size = 32 

def create_dataset(files):
    image_paths = []
    labels = []
    for file in files:
        df = pd.read_csv(file)
        image_paths.extend(df['file'].tolist())
        labels.extend(df['year'].tolist())
    return image_paths, labels

def load_and_preprocess_image(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, image_size)
    image = image / 255.0
    return image, label

def get_tf_dataset(files):
    image_paths, labels = create_dataset(files)
    image_paths = tf.constant(image_paths)
    labels = tf.constant(labels)
    ds = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    ds = ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.shuffle(buffer_size=len(image_paths))
    #ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

train_ds = get_tf_dataset(fold_files)
val_ds = get_tf_dataset(test_files)

print(train_ds)
print(val_ds)


base_model = NASNetLarge(
    include_top=True,
    weights="imagenet",
    classifier_activation="softmax",
)

x = base_model.layers[-1].output
x = Flatten(name='FLATTEN')(x)
x = Dense(256, activation='relu', name='last_FC1')(x)
x = Dropout(0.5, name='DROPOUT')(x)
predictions = Dense(len(classes), activation='softmax', name='PREDICTIONS')(x)
model = Model(input=base_model.input, outputs=predictions)

for layer in model.layers[:]:
    layer.trainable = False

model.compile(optimizer=Adam(lr=0.0001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

model.summary()
