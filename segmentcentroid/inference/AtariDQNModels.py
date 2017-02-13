"""
This module implements a deep q network in tensorflow
"""
import tensorflow as tf
import numpy as np
import copy
from .DQN import DQN

class RegressionDQN(DQN):

    def __init__(self,
                 env,
                 statedim,
                 actiondim,
                 hidden_layer=64,
                 buffersize = 100000,
                 gamma = 0.99,
                 learning_rate = 0.1,
                 minibatch=100,
                 epsilon0=1,
                 epsilon_decay_rate=1e-3):

        self.hidden_layer = hidden_layer
        super(RegressionDQN, self).__init__(env, statedim, actiondim, buffersize, gamma, learning_rate, minibatch, epsilon0, epsilon_decay_rate)


    def createQNetwork(self):

        x = tf.placeholder(tf.float32, shape=[None, self.statedim[0], 1])

        xp = tf.reshape(x, (-1, self.statedim[0]))

        a = tf.placeholder(tf.float32, shape=[None, self.actiondim])

        y = tf.placeholder(tf.float32, shape=[None, 1])

    
        W_1 = tf.Variable(tf.random_normal([self.statedim[0], self.hidden_layer ]))
        b_1 = tf.Variable(tf.random_normal([self.hidden_layer]))

        h1 = tf.nn.sigmoid(tf.matmul(xp, W_1) + b_1)

        W_2 = tf.Variable(tf.random_normal([self.hidden_layer, self.hidden_layer]))
        b_2 = tf.Variable(tf.random_normal([self.hidden_layer]))

        h2 = tf.nn.sigmoid(tf.matmul(h1, W_2) + b_2)

        W_3 = tf.Variable(tf.random_normal([self.hidden_layer, self.actiondim ]))
        b_3 = tf.Variable(0.0*tf.random_normal([self.actiondim ]))

        alloutput = tf.matmul(h2, W_3) + b_3

        output = tf.reshape(tf.reduce_mean(tf.multiply(a, alloutput), 1), [-1, 1])

        woutput = tf.reduce_mean((y-output)**2)
        
        return {'state': x, 
                'action': a, 
                'y': y,
                'output': output,
                'alloutput': alloutput,
                'woutput': woutput}


    def observe(self, s):
        sp = np.zeros(shape=[1, self.statedim[0], 1])
        s = s.reshape((self.statedim[0], 1))
        sp[0,:] = s
        return sp










