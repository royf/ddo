from .TFLayeredModel import TFLayeredModel
from .unsupervised_vision_networks import *
from .supervised_networks import *
import tensorflow as tf
import numpy as np

class JHUJigSawsMultimodalModel(TFLayeredModel):
    
    """
    This class defines the abstract class for a tensorflow model for the primitives.
    """

    def __init__(self,  
                 k,
                 statedim=(120, 160, 3), 
                 actiondim=(37,1),
                 hidden_layer=64,
                 variance=10000):

        self.hidden_layer = hidden_layer
        self.variance = variance
        
        super(JHUJigSawsMultimodalModel, self).__init__(statedim, (64,1) , actiondim, k)


    def createPolicyNetwork(self):

        #return affine(self.statedim[0],
        #              self.actiondim[0],
        #              self.variance)  

        return affine(self.statedim[0],
                      self.actiondim[0],
                      variance=self.variance) 

    def createTransitionNetwork(self):

        return multiLayerPerceptron(self.statedim[0],
                                    2,
                                    self.hidden_layer)


    def createUnsupervisedNetwork(self):
        return twoLayerAutoencoder([120, 160, 3], 37)


    def preloader(self, traj):
        #only video

        return [(t[0][1].astype("float32"), t[1]) for t in traj]




