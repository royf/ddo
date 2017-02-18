"""
This module implements a deep q network in tensorflow
"""
import tensorflow as tf
import numpy as np
import copy
from .DQN import *

class RegressionDQN(DQN):

    def __init__(self,
                 env,
                 statedim,
                 actiondim,
                 hidden_layer=128,
                 buffersize = 100000,
                 gamma = 0.99,
                 learning_rate = 0.1,
                 minibatch=100,
                 epsilon0=1,
                 epsilon_decay_rate=1e-3):

        self.hidden_layer = hidden_layer
        super(RegressionDQN, self).__init__(env, statedim, actiondim, buffersize, gamma, learning_rate, minibatch, epsilon0, epsilon_decay_rate)
        self.checkpoint_frequency = 1000
        self.eval_frequency = 20
        self.eval_trials = 10



    def createQNetwork(self):

        x = tf.placeholder(tf.float32, shape=[None, 128])

        a = tf.placeholder(tf.float32, shape=[None, self.actiondim])

        y = tf.placeholder(tf.float32, shape=[None, 1])
    
        W_1 = tf.Variable(tf.random_normal([128, self.hidden_layer]))

        h1 = tf.nn.relu(tf.matmul(x, W_1))


        W_2 = tf.Variable(tf.random_normal([128, self.hidden_layer]))

        h2 = tf.nn.relu(tf.matmul(h1, W_2))

        W_3 = tf.Variable(tf.random_normal([self.hidden_layer, self.actiondim]))
        b_3 = tf.Variable(0*tf.random_normal([self.actiondim]))

        alloutput = tf.matmul(h1, W_3) + b_3

        output = tf.reshape(tf.reduce_mean(tf.multiply(a, alloutput), 1), [-1, 1])

        woutput = tf.reduce_mean((y-output)**2)
        
        return {'state': x, 
                'action': a, 
                'y': y,
                'output': output,
                'debug': (y-output)**2,
                'alloutput': alloutput,
                'woutput': woutput}


    def observe(self, s):
        sp = np.zeros(shape=[128, 1])
        sp = np.reshape(s, (1,128))
        return sp/256.0



class AtariTabularDQN(TabularDQN):

    def __init__(self,
                 env,
                 statedim,
                 actiondim,
                 buffersize = 100000,
                 gamma = 0.99,
                 learning_rate = 0.1,
                 minibatch=100,
                 epsilon0=1,
                 epsilon_decay_rate=1e-3):

        super(AtariTabularDQN, self).__init__(env, statedim, actiondim, buffersize, gamma, learning_rate, minibatch, epsilon0, epsilon_decay_rate)

        self.checkpoint_frequency = 1000
        self.eval_frequency = 10
        self.eval_trials = 1


    def createQNetwork(self):
        sarraydims = copy.copy(self.sarraydims)
        sarraydims.insert(0, None)

        x = tf.placeholder(tf.float32, shape=sarraydims)

        a = tf.placeholder(tf.float32, shape=[None, self.actiondim])

        y = tf.placeholder(tf.float32, shape=[None, 1])

    
        table = tf.Variable(tf.random_uniform([1, self.statedim[0], self.actiondim]))

        inputx = tf.tile(tf.reshape(x, [-1, self.statedim[0], 1]), [1, 1, self.actiondim])

        tiled_table = tf.tile(table, [tf.shape(x)[0],1,1])

        collapse = tf.reshape(tf.reduce_sum(tf.multiply(inputx, tiled_table), 1), [-1, self.actiondim])

        output = tf.reshape(tf.reduce_mean(tf.multiply(a, collapse), 1), [-1, 1])

        woutput = tf.reduce_mean((y-output)**2)
        
        return {'state': x, 
                'action': a, 
                'y': y,
                'output': output,
                'alloutput': collapse,
                'woutput': woutput}


    def observe(self, s):
        sarraydims = copy.copy(self.sarraydims)
        sarraydims.insert(0, 1)

        sp = np.zeros(shape=sarraydims)
        sp[0, s] =  1
        return sp


class LinearDQN(DQN):

    def __init__(self,
                 env,
                 statedim,
                 actiondim,
                 hidden_layer=32,
                 regularization=0.01,
                 buffersize = 100000,
                 gamma = 0.99,
                 learning_rate = 0.1,
                 minibatch=100,
                 epsilon0=1,
                 epsilon_decay_rate=1e-3):

        self.hidden_layer = hidden_layer
        self.regularization = regularization

        super(LinearDQN, self).__init__(env, statedim, actiondim, buffersize, gamma, learning_rate, minibatch, epsilon0, epsilon_decay_rate)
        self.checkpoint_frequency = 1000
        self.eval_frequency = 10
        self.eval_trials = 10
        self.update_frequency = 2



    def createQNetwork(self):

        x = tf.placeholder(tf.float32, shape=[None, self.statedim[0]])

        a = tf.placeholder(tf.float32, shape=[None, self.actiondim])

        y = tf.placeholder(tf.float32, shape=[None, 1])

        W1 = tf.Variable(tf.random_normal([self.statedim[0],  self.hidden_layer]))
        b1 = tf.Variable(tf.random_normal([ self.hidden_layer]))

        h = tf.nn.relu(tf.matmul(x, W1) + b1)

        W2 = tf.Variable(tf.random_normal([ self.hidden_layer, self.actiondim]))
        b2 = tf.Variable(tf.random_normal([self.actiondim]))

        alloutput = tf.reshape(tf.matmul(h, W2) + b2, [-1, self.actiondim])

        output = tf.reshape(tf.reduce_mean(tf.multiply(a, alloutput), 1), [-1, 1])

        weights = [W1, b1, W2, b2]

        woutput = tf.reduce_mean(tf.square(y-output)) #+  0.1*(tf.reduce_sum(tf.square(W)) + tf.reduce_sum(tf.square(b)))

        #for w in weights:
        woutput = woutput + self.regularization*(tf.reduce_sum(tf.square(alloutput)))
        
        return {'state': x, 
                'action': a, 
                'y': y,
                'output': output,
                'debug': h,
                'weights': weights,
                'alloutput': alloutput,
                'woutput': woutput}


    def observe(self, s):
        sp = np.reshape(s, (1,self.statedim[0]))
        return sp