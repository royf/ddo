
from .AbstractModel import AbstractModel
import numpy as np
import scipy.special

from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import NotFittedError

import tensorflow as tf

"""
Defines a linear logistic model
"""


class TFModel(AbstractModel):

    def __init__(self,statedim, actiondim, unnormalized=False):
      
        #self.theta = 10*np.random.rand(statedim, actiondim)

        self.x, self.a, self.y, self.logprob = self.initialize_network(statedim, actiondim)
        self.sess = tf.Session()
        self.sess.run(tf.initialize_all_variables())
        self.isTabular = True

        super(TFModel, self).__init__(statedim, actiondim, discrete=True, unnormalized=unnormalized)


    def initialize_network(self, 
                           statedim, 
                           actiondim, 
                           hidden=512):

        x = tf.placeholder(tf.float32, shape=[None, statedim])

        #must be one-hot encoded
        a = tf.placeholder(tf.float32, shape=[None, actiondim])

        W_h1 = tf.Variable(tf.random_normal([statedim, hidden]))
        b_1 = tf.Variable(tf.random_normal([hidden]))
        h1 = tf.nn.sigmoid(tf.matmul(x, W_h1) + b_1)

        W_out = tf.Variable(tf.random_normal([hidden, actiondim]))
        b_out = tf.Variable(tf.random_normal([actiondim]))
        y = tf.nn.softmax(tf.matmul(h1, W_out) + b_out)

        logprob = tf.reduce_sum(- a * tf.log(y), 1)

        return x, a, y, logprob


    #returns a probability distribution over actions
    def eval(self, s):
        feed_dict = {self.x: s}
        return self.sess.run(self.y, feed_dict)

    #return the log derivative log \nabla_\theta \pi(s)
    def log_deriv(self, s, a):

        encoded_action = np.zeros((1,self.actiondim))
        encoded_action[0,a] = 1
        feed_dict = {self.x: s, self.a: encoded_action}
        grads = tf.gradients(self.logprob, tf.trainable_variables())
        grads = list(zip(grads, tf.trainable_variables()))
        
        result = []
        for i in grads:
            if i[0] != None:
                result.append((i[1], self.sess.run(i[0],feed_dict)))

        return result

    def descent(self, grad_theta, learning_rate):

        for i,g in enumerate(grad_theta):
            weight = g[0]
            print(i)

            for v in g[1]:
                tfvar = v[0]
                grad = v[1]
                self.sess.run(tfvar.assign(tfvar - weight*learning_rate*grad))




