import tensorflow.keras.backend as K
from PIL import Image
from tensorflow.keras.datasets import mnist
from tensorflow.keras.datasets import cifar10
import tensorflow as tf
import numpy as np
import cv2

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

    for i in range(x_train.shape[0]):
        X_train[i] = cv2.cvtColor(x_train[i,:,:,0],cv2.COLOR_GRAY2RGB)

    for i in range(x_test.shape[0]):
        X_test[i] = cv2.cvtColor(x_test[i,:,:,0],cv2.COLOR_GRAY2RGB)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

    # convert class vectors to binary class matrices
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)
    ## MNIST dataset prepared

    return (X_train, y_train), (X_test, y_test)

def load_cifar():
    img_rows, img_cols = 32, 32
    num_classes = 1000

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    
    X_train = np.zeros((x_train.shape[0],x_train.shape[1],x_train.shape[2],3)).astype('uint8')
    X_test = np.zeros((x_test.shape[0],x_test.shape[1],x_test.shape[2],3)).astype('uint8')
    
    for i in range(x_train.shape[0]):
        X_train[i] = cv2.cvtColor(x_train[i,:,:,0],cv2.COLOR_GRAY2RGB)

    for i in range(x_test.shape[0]):
        X_test[i] = cv2.cvtColor(x_test[i,:,:,0],cv2.COLOR_GRAY2RGB)
    
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)
    
    return (X_train, y_train), (X_test, y_test)


def load(dataset, concealing, c_size):
    cover = None
    if concealing:
        path =  "./data/dog.jpg"
        imgnet = cv2.imread(path)
        cover = np.array(cv2.resize(imgnet, dsize=c_size, interpolation=cv2.INTER_CUBIC), dtype='float32')
        cover /= 255
        cover = np.array([cover])
    if dataset == 'MNIST':
        load_dataset = load_mnist
    elif dataset == 'CIFAR10':
        load_dataset = load_cifar
    else:
        load_dataset = None
    train, test = load_dataset()
    return train, test, cover
