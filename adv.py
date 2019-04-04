import tensorflow as tf
import tensorflow.contrib.eager as tfe
tf.enable_eager_execution()

import numpy as np
import data_utils

class AdversarialProgramming(tf.keras.Model):
    def __init__(self, W, M, padding, adv_size, alpha):
        super(AdversarialProgramming, self).__init__()
        self.W = tfe.Variable(W)
        self.M = M
        self.padding = padding
        (self.h, self.w) = adv_size
        self.alpha = alpha

    def call(self, inputs):
        ps = np.empty(shape=[inputs.shape[0], self.h, self.w, 3])
        ps = [tf.tanh(self.W*self.M) for x in ps]
        prog = ps + inputs
        return prog
