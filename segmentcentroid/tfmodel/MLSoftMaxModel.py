from .TFModel import TFModel
import tensorflow as tf
import numpy as np

class MLSoftMaxModel(TFModel):
    
    """
    This class defines the abstract class for a tensorflow model for the primitives.
    """

    def __init__(self, 
                 statedim, 
                 actiondim, 
                 k,
                 hidden_layer=512):

        self.hidden_layer = hidden_layer
        self.policy_networks = []
        self.transition_networks = []
        
        super(MLSoftMaxModel, self).__init__(statedim, actiondim, k)


    def createPolicyNetwork(self):

        if self.statedim[1] != 1:
            raise ValueError("MLSoftMaxModels only apply to vector valued state-spaces")

        if self.actiondim[1] != 1:
            raise ValueError("MLSoftMaxModels only apply to vector valued (1 hot encoded) action-spaces")

        sdim = self.statedim[0]
        adim = self.actiondim[0]

        x = tf.placeholder(tf.float32, shape=[None, sdim])

        #must be one-hot encoded
        a = tf.placeholder(tf.float32, shape=[None, adim])

        #must be a scalar
        weight = tf.placeholder(tf.float32, shape=[None, 1])

        W_h1 = tf.Variable(tf.random_normal([sdim, self.hidden_layer]))
        b_1 = tf.Variable(tf.random_normal([self.hidden_layer]))
        h1 = tf.nn.sigmoid(tf.matmul(x, W_h1) + b_1)

        W_out = tf.Variable(tf.random_normal([self.hidden_layer, adim]))
        b_out = tf.Variable(tf.random_normal([adim]))
        
        logit = tf.matmul(h1, W_out) + b_out
        y = tf.nn.softmax(logit)

        logprob = -tf.log(tf.reduce_sum(y*a))

        wlogprob = weight*logprob
        
        return {'state': x, 
                'action': a, 
                'weight': weight,
                'prob': y, 
                'lprob': logprob,
                'wlprob': wlogprob}

    def createTransitionNetwork(self):

        if self.statedim[1] != 1:
            raise ValueError("MLSoftMaxModels only apply to vector valued state-spaces")

        sdim = self.statedim[0]
        adim = 2 #binary

        x = tf.placeholder(tf.float32, shape=[None, sdim])

        #must be one-hot encoded
        a = tf.placeholder(tf.float32, shape=[None, adim])

        #must be a scalar
        weight = tf.placeholder(tf.float32, shape=[None, 1])

        W_h1 = tf.Variable(tf.random_normal([sdim, self.hidden_layer]))
        b_1 = tf.Variable(tf.random_normal([self.hidden_layer]))
        h1 = tf.nn.sigmoid(tf.matmul(x, W_h1) + b_1)

        W_out = tf.Variable(tf.random_normal([self.hidden_layer, adim]))
        b_out = tf.Variable(tf.random_normal([adim]))
        logit = tf.matmul(h1, W_out) + b_out

        y = tf.nn.softmax(logit)
        logprob = -tf.log(tf.reduce_sum(y*a))

        wlogprob = weight*logprob
        
        return {'state': x, 
                'action': a, 
                'weight': weight,
                'prob': y, 
                'lprob': logprob,
                'wlprob': wlogprob}


    def getLossFunction(self):

        loss_array = []

        pi_vars = []
        for i in range(0, self.k):
            loss_array.append(self.policy_networks[i]['wlprob'])
            pi_vars.append((self.policy_networks[i]['state'], 
                            self.policy_networks[i]['action'], 
                            self.policy_networks[i]['weight']))

        psi_vars = []
        for i in range(0, self.k):
            loss_array.append(self.transition_networks[i]['wlprob'])
            psi_vars.append((self.transition_networks[i]['state'], 
                            self.transition_networks[i]['action'], 
                            self.transition_networks[i]['weight']))

        return tf.reduce_sum(loss_array), pi_vars, psi_vars



    def initialize(self):
        for i in range(0, self.k):
            self.policy_networks.append(self.createPolicyNetwork())

        for i in range(0, self.k):
            self.transition_networks.append(self.createTransitionNetwork())

        self.sess.run(tf.initialize_all_variables())


    #returns a probability distribution over actions
    def _evalpi(self, index, s, a):
        feed_dict = {self.policy_networks[index]['state']: s.reshape((1, self.statedim[0]))}
        encoded_action = np.argwhere(a > 0)[0][0]
        dist = np.ravel(self.sess.run(self.policy_networks[index]['prob'], feed_dict))
        
        if np.isnan(dist[0]):
            print(dist,s, a, self.sess.run(self.policy_networks[index]['prob'], feed_dict))

            for v in tf.trainable_variables():
                print("Dumping variables")
                print(v, self.sess.run(v))

            raise ValueError("Error!!!")

        #print(s, np.sum(dist), dist)

        #if np.sum(dist) == 0:
        #    print("clip")
        #    dist = np.ones(dist.shape)

        return dist[encoded_action]/np.sum(dist)

    #returns a probability distribution over actions
    def _evalpsi(self, index, s):
        feed_dict = {self.transition_networks[index]['state']: s.reshape((1, self.statedim[0]))}
        encoded_action = 1
        dist = np.ravel(self.sess.run(self.transition_networks[index]['prob'], feed_dict))
        
        #print(s, np.sum(dist), dist)

        #if np.sum(dist) == 0:
        #    print("clip")
        #    dist = np.ones(dist.shape)

        return dist[encoded_action]/np.sum(dist)
