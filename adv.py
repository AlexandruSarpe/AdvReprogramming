import tensorflow as tf
import tensorflow.contrib.eager as tfe
tf.enable_eager_execution()

import numpy as np
import data_utils

class AdversarialProgramming(tf.keras.Model):
    def __init__(self, W, M, adv_size, alpha):
        super(AdversarialProgramming, self).__init__()
        self.W = tfe.Variable(W)
        self.M = M
        (self.h, self.w) = adv_size
        self.alpha = alpha

    def call(self, inputs):
        ps = np.empty(shape=[inputs.shape[0], self.h, self.w, 3])
        ps = [tf.tanh(self.W*self.M) for x in ps]
        prog = ps + inputs
        return prog

def loadTargetNet(net):
    if net == 'Inception_V3':
        from tensorflow.keras.applications import InceptionV3
        return (299, 299), InceptionV3()
    else:
        return 0, 0, None

def computePadding(target_size, in_size):
    '''
    Computes padding to a target size
    from a given in_size
    '''
    target_h, target_w = target_size
    in_h, in_w = in_size
    x_padding = int(np.floor((target_h-in_h)/2))
    y_padding = int(np.ceil((target_w-in_w)/2))
    padding = ([x_padding, y_padding], [x_padding, y_padding])
    return padding

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', type=int, default=2,
        help='number of epochs')
    parser.add_argument('-i', '--infile', type=str, default='weights.npy',
        help='weights load file')
    parser.add_argument('-o', '--outfile', type=str, default='weigths_train.npy',
        help='weigths save file')
    parser.add_argument('-l', '--lambda', type=float, default=0.01,
        help='regularizer')
    parser.add_argument('-a', '--alpha', type=float, default=0.5,
        help='perturbation limiter')
    parser.add_argument('-n', '--net', type=str, default='Inception_V3',
        help='Net to reprogram')
    parser.add_argument('-d', '--dataset', type=str, default='MNIST',
        help='dataset to use')
    parser.add_argument('-b', '--batch_size', type=int, default=32,
        help='size of mini-batches')
    args = vars(parser.parse_args())

    alpha = args['alpha']

    # Load Images and Labels from specific dataset
    (X_train, y_train), (X_test, y_test) = data_utils.load(dataset=args['dataset'])
    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    # Create train and test iterators
    train_it = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    test_it = tf.data.Dataset.from_tensor_slices((X_test, y_test))

    # Load target model to reprogram
    adv_size, target = loadTargetNet(net=args['net'])
    if target == None:
        print('Failed to load target net: %s' % 'Inception_V3')
        sys.exit(-1)
    target.trainable = False
