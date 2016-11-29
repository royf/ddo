import tensorflow as tf
import numpy as np
from segmentcentroid.inference.forwardbackward import ForwardBackward

class TFModel(object):
    
    """
    This class defines the abstract class for a tensorflow model for the primitives.
    """

    def __init__(self, 
                 statedim, 
                 actiondim, 
                 k):
        """
        Defines the state-space and action-space and number of primitves
        """

        self.statedim = statedim #numpy shape
        self.actiondim = actiondim #numpy shape
        self.k = k
        self.fb = None
        self.sess = tf.Session()

        self.initialize()


    def initialize(self):
        raise NotImplemented("Must implement an initialize function")

    def evalpi(self, index, s, a):
        """
        Returns the probability of action a at state s
        """

        if index >= self.k:
            raise ValueError("Primitive index is greater than the number of primitives")

        s = np.reshape(s, self.statedim)

        a = np.reshape(a, self.actiondim)

        return self._evalpi(index, s, a)

    #returns a probability distribution over actions
    def _evalpi(self, index, s, a):
        raise NotImplemented("Must implement an _evalpi function")


    def evalpsi(self, index, s):
        """
        Returns the probability of action a at state s
        """

        if index >= self.k:
            raise ValueError("Primitive index is greater than the number of primitives")

        s = np.reshape(s, self.statedim)

        return self._evalpsi(index, s)

    #returns a probability distribution over actions
    def _evalpsi(self, index, s):
        raise NotImplemented("Must implement an _evalpsi function")


    """
    Fitting primitives
    """

    def fit(self, 
            X, 
            hmm_config={}, 
            gd_config={}):
        """
        X is a list of trajectories
        hmm_config is config that goes to the forward 
        backward algorithm
        gd config is config that goes to tensorflow
        """

        self.fb = ForwardBackward(self, **hmm_config)
        print(self.fb.fit(X))

        
