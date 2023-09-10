# This project is licensed by the MIT License
# Copyright © 2023 RandomKiddo, Nishanth-Kunchala, danield33

import tensorflow as tf
import tensorflow_datasets as tfds
from typing import *
import matplotlib.pyplot as plt
import keras
from keras.layers import *
from keras.losses import SparseCategoricalCrossentropy
from keras.models import Sequential


def load_data() -> tuple[Any, Any]:
    """
    Loads the EMNIST data
    :return: Train and test datasets
    """

    (train_ds, test_ds), info = tfds.load(
        'emnist', split=['train', 'test'], shuffle_files=True, as_supervised=True, with_info=True
    )

    def normalize(image: Any, label: Any) -> tuple[Any, Any]:
        """
        Normalize a given image        
        :param image: The given image
        :param label: The label to the image
        :return: The normalized version
        """
        
        return tf.cast(image, tf.float32) / 255., label
    
    train_ds = train_ds.map(normalize, num_parallel_calls=tf.data.AUTOTUNE).cache().shuffle(
        info.splits['train'].num_examples).batch(32).prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.map(normalize, num_parallel_calls=tf.data.AUTOTUNE).batch(32).cache().prefetch(tf.data.AUTOTUNE)

    return train_ds, test_ds


def create_model(verbose: bool = False) -> None:
    """
    Creates the tensorflow model
    :return: None
    """

    train_ds, test_ds = load_data()

    data_augmentation = Sequential([
        RandomFlip('horizontal', input_shape=(28, 28, 1)),
        RandomRotation(0.1),
        RandomZoom(0.1),
        RandomContrast(0.1)
    ])

    model = Sequential([
        data_augmentation,
        Conv2D(16, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Conv2D(32, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Conv2D(64, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Dropout(0.2),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(62)
    ])

    model.compile(optimizer='Adam', loss=SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    if verbose:
        model.summary()

    epochs = 30
    history = model.fit(train_ds, epochs=epochs)

    acc = history.history['accuracy']
    loss = history.history['loss']
    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.title('Training Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.title('Training Loss')

    plt.savefig('metrics.png')
    plt.show()


if __name__ == '__main__':
    create_model(verbose=True)

