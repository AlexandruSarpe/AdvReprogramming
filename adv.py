import tensorflow as tf
import tensorflow.contrib.eager as tfe
tf.enable_eager_execution()

import matplotlib.pyplot as plt
import numpy as np
import data_utils
import argparse
import time
import sys


class AdversarialProgramming(tf.keras.Model):
    def __init__(self, W, M, adv_size, alpha, cover):
        super(AdversarialProgramming, self).__init__()
        self.W = tfe.Variable(W)
        self.M = M
        (self.h, self.w) = adv_size
        self.alpha = alpha
        self.cover = cover

    def call(self, inputs):
        if self.cover is None:
            ps = tf.tanh(self.W*self.M)
            prog = ps + inputs
        else:
            ps = self.cover + self.alpha*tf.tanh(inputs + self.W*self.M)
            prog = tf.clip_by_value(ps, 0, 1)
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
    #return tf.random_uniform(shape = [h_image, w_image, 3])
    return np.zeros(shape=[1, h_image, w_image, 3], dtype='float32')

def createMask(size, pad):
    '''
    Computes mask
    '''
    height, width = size
    (i_min, i_max), (j_min, j_max) = pad
    M = np.ones((height, width, 3)).astype('float32')
    M[i_min:height-i_max-1, j_min:width-j_max-1, :] = 0
    return np.array([M])

def shuffle_custom(mat, permx, permy):
    new_ = np.empty(mat.shape)
    tmp_ = np.empty(mat.shape)
    for e,i in zip(permx, range(mat.shape[1])):
        tmp_[:,i] = mat[:,e]
    for e,i in zip(permy, range(mat.shape[2])):
        new_[:,:,i] = tmp_[:,:,e]
    return new_

def label_mapping():
    imagenet_label = np.zeros([1000,10])
    imagenet_label[0:10,0:10]=np.eye(10)
    return tf.constant(imagenet_label, dtype=tf.float32) 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', type=int, default=2,
        help='number of epochs')
    parser.add_argument('-i', '--infile', type=str, default='weights.npy',
        help='weights load file')
    parser.add_argument('-o', '--outfile', type=str, default='weigths_train.npy',
        help='weigths save file')
    parser.add_argument('-l', '--lambda', type=float, default=5e-4,
        help='regularizer')
    parser.add_argument('-a', '--alpha', type=float, default=0.5,
        help='perturbation limiter')
    parser.add_argument('-n', '--net', type=str, default='Inception_V3',
        help='Net to reprogram')
    parser.add_argument('-d', '--dataset', type=str, default='MNIST',
        help='dataset to use')
    parser.add_argument('-b', '--batch_size', type=int, default=32,
        help='size of mini-batches')
    parser.add_argument('-L', '--load', type=bool, default=False,
        help='Load model wheights from file')
    parser.add_argument('-c', '--concealing', type=bool, default=False,
        help='Use concealing')
    args = vars(parser.parse_args())

    LOAD = args['load']
    alpha = args['alpha']

    # Load target model to reprogram
    adv_size, target = loadTargetNet(net=args['net'])
    if target == None:
        print('Failed to load target net: %s' % 'Inception_V3')
        sys.exit(-1)
    target.trainable = False

    # Load Images and Labels from specific dataset
    (X_train, y_train), (X_test, y_test), cover = data_utils.load(dataset=args['dataset'], concealing=args['concealing'], c_size=adv_size)
    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')
    
    in_size = (X_train[0].shape[0], X_train[0].shape[1])
    padding = computePadding(adv_size, in_size)
   
    # Create train and test iterators
    train_it = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    test_it = tf.data.Dataset.from_tensor_slices((X_test, y_test))

    M = tf.constant(createMask(adv_size, padding))
    W = createW(adv_size)
    
    if args['concealing']:
        permx = np.arange(adv_size[0])
        permy = np.arange(adv_size[1])

        np.random.shuffle(permx)
        np.random.shuffle(permy)
        M = shuffle_custom(M, permx, permy)

    cce = tf.keras.losses.CategoricalCrossentropy()
    #cce = tf.nn.softmax_cross_entropy_with_logits

    #target = inception_v3.inception_v3

    def loss(model, xb, yb):
        '''
        We define our loss function as:
        -Log(P(yb|Xadv))+lambda*||W||^2
        '''
        adv = model(tf.cast(xb, dtype='float32'))
        y_pred = target(adv)
        y_pred = tf.matmul(y_pred, label_mapping())
        yb = tf.matmul(yb, label_mapping())
        return tf.reduce_mean(cce(yb, y_pred))+args['lambda']*tf.nn.l2_loss(model.W)

    adv_model = AdversarialProgramming(W, M, adv_size, alpha, cover)

    global_steps = tf.Variable(0, trainable=False)
    steps_per_epoch = X_train.shape[0]/args['batch_size']
    decay_steps = 10 * steps_per_epoch
    lr = tf.train.exponential_decay(0.05, global_steps, decay_steps, 0.96, staircase=True)
    opt = tf.train.AdamOptimizer(learning_rate=lr)

    # Create validation set
    b_val = 25
    data = tf.pad(X_test[:b_val], [[0,0],padding[0],padding[1],[0,0]])
    if args['concealing']:
        data = shuffle_custom(data, permx, permy)
    labels = np.array(y_test[:b_val])
    labels = tf.matmul(labels, label_mapping())
    
    if not LOAD:
        for epoch in range(args['epochs']):
            tick = time.time()
            print("Epoch: {}".format(epoch+1))
            i = 0
            for _ in range(20):
                for xb, yb in train_it.batch(args['batch_size']):
                    '''
                    every 50 steps we compute the current accuracy
                    '''
                    if i % 50 == 0:
                        if not args['concealing']:
                            ps = tf.tanh(adv_model.W*M)
                            prog = ps + data
                        else:
                            prog = np.clip(cover+alpha*tf.tanh(data + adv_model.W*M), 0, 1)
                        preds = target(prog)
                        preds = tf.matmul(preds, label_mapping())
                        count = 0
                        for j in range(labels.shape[0]):
                            if np.argmax(preds[j].numpy()) == np.argmax(labels[j]):
                                count += 1
                        print("     Test acc at step {}: {}".format(i+1, count/b_val))
                    i+=1
                    '''
                    minimize the loss using Adam with a decayed learing rate of 0.05
                    '''
                    xb = tf.pad(xb, [[0,0],padding[0],padding[1],[0,0]])
                    if args['concealing']:
                        xb = shuffle_custom(xb, permx, permy)
                    opt.minimize(lambda: loss(adv_model, xb, yb), var_list=[adv_model.W], global_step=global_steps)
                plt.imshow(adv_model.W.numpy()[0])
                plt.show() 
                elapsed = time.time()-tick
                print("elapsed time: {} seconds".format(elapsed))

        '''
        save weigths into file
        '''
        with open(args['outfile'], "wb") as f:
            np.save(f, adv_model.W.numpy())
    else:
        acc_log = []
        for data, labels in test_it.batch(b_val):
            data = tf.pad(data, [[0,0],padding[0],padding[1],[0,0]])
            if not args['concealing']:
                ps = tf.tanh(W*M)
                prog = ps + data 
            else:
                data = shuffle_custom(data, permx, permy)
                ps = cover + alpha*tf.tanh(data + W*M)
                prog = tf.clip_by_value(ps, 0, 1)
            preds = target(tf.cast(prog, dtype='float32'))
            count = 0
            for j in range(data.shape[0]):
                if np.argmax(preds[j].numpy()) == np.argmax(labels[j]):
                    count += 1
            acc_log.append(count/b_val)
        print("Non trained weights acc {}".format(np.mean(acc_log)))

        W = np.load(args['infile'])
        plt.imshow(W[0])
        plt.show()
        acc_log = []
        for data, labels in test_it.batch(b_val):
            data = tf.pad(data, [[0,0],padding[0],padding[1],[0,0]])
            if not args['concealing']:
                ps = tf.tanh(W*M)
                prog = ps + data 
            else:
                data = shuffle_custom(data, permx, permy)
                ps = cover + alpha*tf.tanh(data + W*M)
                prog = tf.clip_by_value(ps, 0, 1)
            preds = target(tf.cast(prog, dtype='float32'))
            count = 0
            for j in range(data.shape[0]):
                if np.argmax(preds[j].numpy()) == np.argmax(labels[j]):
                    count += 1
            acc_log.append(count/b_val)
        print("Trained weights acc {}".format(np.mean(acc_log)))
