import tensorflow as tf
import tensorflow.contrib.eager as tfe
tf.enable_eager_execution()

import numpy as np
import data_utils
import time
import sys

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

def createW(size):
    '''
    Inits Weights following a random uniform curve
    '''
    h_image, w_image = size
    return tf.random_uniform(shape = [h_image, w_image, 3])

def createMask(size, pad):
    '''
    Computes mask
    '''
    height, width = size
    (i_min, i_max), (j_min, j_max) = pad
    M = np.ones((height, width, 3)).astype('float32')
    M[i_min:height-i_max-1, j_min:width-j_max-1, :] = 0
    return M

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

    in_size = (X_train[0].shape[0], X_train[0].shape[1])
    padding = computePadding(adv_size, in_size)

    M = tf.constant(createMask(adv_size, padding))
    W = createW(adv_size)

    cce = tf.keras.losses.CategoricalCrossentropy()

    def loss(model, xb, yb):
        '''
        We define our loss function as:
        argmin(-Log(P(yb|Xadv))+lambda*||W||^2)
        '''
        adv = model(xb)
        y_pred = target(adv)
        return tf.reduce_min(cce(yb, y_pred))+args['lambda']*tf.nn.l2_loss(W)

    adv_model = AdversarialProgramming(W, M, adv_size, alpha)
    opt = tf.train.AdamOptimizer(learning_rate=0.05)

    # Create validation set
    data = tf.pad(np.array(X_test[:25]), [[0,0],padding[0],padding[1],[0,0]])
    labels = np.array(y_test[:25])

    for epoch in range(args['epochs']):
        tick = time.time()
        print("Epoch: {}".format(epoch+1))
        i = 0
        for xb, yb in train_it.batch(args['batch_size']):
            '''
            every 50 steps we compute the current accuracy
            '''
            if i % 50 == 0:
                ps = np.empty(shape=[data.shape[0], adv_size[0], adv_size[1], 3])
                ps = [tf.tanh(adv_model.W*M) for x in ps]
                prog = ps + data
                preds = target(prog)
                count = 0
                for j in range(data.shape[0]):
                    if np.argmax(preds[j].numpy()) == np.argmax(labels[j]):
                        count += 1
                print("     Test acc at step {}: {}".format(i+1, count/25))
            i+=1
            '''
            minimize the loss using Adam with a learing rate of 0.05
            '''
            padded = tf.pad(xb, [[0,0],padding[0],padding[1],[0,0]])
            opt.minimize(lambda: loss(adv_model, padded, yb), var_list=[adv_model.W])
        elapsed = time.time()-tick
        print("elapsed time: {} seconds".format(elapsed))

    '''
    save weigths into file
    '''
    with open(args['outfile'], "wb") as f:
        np.save(f, adv_model.W.numpy())