import numpy as np
import tensorflow as tf


default_path = 'C:/Users/DimKa/Documents/'


def data(path=default_path):
    log_root = path
    (im_train, y_train), (im_test, y_test) = tf.keras.datasets.cifar10.load_data()
    # Normalize to 0-1 range and subtract mean of training pixels
    im_train = im_train / 255
    im_test = im_test / 255

    mean_training_pixel = np.mean(im_train, axis=(0, 1, 2))
    x_train = im_train - mean_training_pixel
    x_test = im_test - mean_training_pixel

    image_shape = x_train[0].shape
    labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    return x_train, y_train, x_test, y_test, image_shape, log_root, labels
