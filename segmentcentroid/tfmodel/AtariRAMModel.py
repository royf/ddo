from .TFSeparableModel import TFSeparableModel
from .supervised_networks import *
import tensorflow as tf
import numpy as np
import ray

class AtariRAMModel(TFSeparableModel):
    
    """
    This class defines the abstract class for a tensorflow model for the primitives.
    """

    def __init__(self,  
                 k,
                 statedim=(128,256), 
                 actiondim=(8,1),
                 hiddenLayerSize=32):
        
        self.hiddenLayerSize = hiddenLayerSize
        super(AtariRAMModel, self).__init__(statedim, actiondim, k, [0,0],'all')


    def createPolicyNetwork(self):

        #ram inputs in the gym are scaled to 0,1 so don't forget to scale

        # (N x 128 x 256)
        x = tf.placeholder(tf.float32, shape=[None, self.statedim[0], self.statedim[1]])
        a = tf.placeholder(tf.float32, shape=[None, self.actiondim[0]])
        weight = tf.placeholder(tf.float32, shape=[None, 1])

        # (256 x 32)
        w_dense_1 = tf.Variable(tf.random_normal([self.statedim[1], self.hiddenLayerSize]))

        # (N x 128 x 32)
        dense_1 = tf.nn.sigmoid(tf.matmul(x, w_dense_1))

        # (N x 1 x 32) -> (N x 32)
        pool_1 = tf.reshape(tf.reduce_sum(dense_1, 1), [-1, self.hiddenLayerSize])

        # (32 x 32)
        w_dense_2 = tf.Variable(tf.random_normal([self.hiddenLayerSize, self.hiddenLayerSize]))

        #(N x 32)
        dense_2 = tf.nn.sigmoid(tf.matmul(pool_1, w_dense_2))

        # (32 x a)
        output_w = tf.Variable(tf.random_normal([self.hiddenLayerSize, self.actiondim[0]]))

        #output
        logit = tf.matmul(dense_2, output_w)

        y = tf.nn.softmax(logit)

        logprob = tf.reshape(tf.nn.softmax_cross_entropy_with_logits(logit, a), [-1,1])

        wlogprob = tf.multiply(weight, logprob)
            
        return {'state': x, 
                    'action': a, 
                    'weight': weight,
                    'prob': y, 
                    'amax': tf.argmax(y, 1),
                    'lprob': logprob,
                    'wlprob': wlogprob,
                    'discrete': True}


    def createTransitionNetwork(self):
        #ram inputs in the gym are scaled to 0,1 so don't forget to scale

        # (N x 128 x 256)
        x = tf.placeholder(tf.float32, shape=[None, self.statedim[0], self.statedim[1]])
        a = tf.placeholder(tf.float32, shape=[None, 1])
        weight = tf.placeholder(tf.float32, shape=[None, 1])

        # (256 x 32)
        w_dense_1 = tf.Variable(tf.random_normal([self.statedim[1], self.hiddenLayerSize]))

        # (N x 128 x 32)
        dense_1 = tf.nn.sigmoid(tf.matmul(x, w_dense_1))

        # (N x 1 x 32) -> (N x 32)
        pool_1 = tf.reshape(tf.reduce_sum(dense_1, 1), [-1, self.hiddenLayerSize])

        # (32 x 32)
        w_dense_2 = tf.Variable(tf.random_normal([self.hiddenLayerSize, self.hiddenLayerSize]))

        #(N x 32)
        dense_2 = tf.nn.sigmoid(tf.matmul(pool_1, w_dense_2))

        # (32 x 1)
        output_w = tf.Variable(tf.random_normal([self.hiddenLayerSize, 1]))

        #output
        logit = tf.matmul(dense_2, output_w)

        y = tf.nn.softmax(logit)

        logprob = tf.reshape(tf.nn.softmax_cross_entropy_with_logits(logit, a), [-1,1])

        wlogprob = tf.multiply(weight, logprob)
            
        return {'state': x, 
                    'action': a, 
                    'weight': weight,
                    'prob': y, 
                    'amax': tf.argmax(y, 1),
                    'lprob': logprob,
                    'wlprob': wlogprob,
                    'discrete': True}




