import tensorflow.keras.backend as K
from PIL import Image
from tensorflow.keras.datasets import mnist
import tensorflow as tf
import numpy as np

def load_mnist():
    # the data, split between train and test sets
    img_rows, img_cols = 28, 28
    num_classes = 1000

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (3, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 3)


    X_train = np.zeros((x_train.shape[0],x_train.shape[1],x_train.shape[2],3)).astype('uint8')
    X_test = np.zeros((x_test.shape[0],x_test.shape[1],x_test.shape[2],3)).astype('uint8')

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

    # convert class vectors to binary class matrices
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)
    ## MNIST dataset prepared

    return (X_train, y_train), (X_test, y_test)