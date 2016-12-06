from .TFModel import TFNetworkModel
from .models import *
import tensorflow as tf
import numpy as np

class JHUJigSawsModel(TFNetworkModel):
    
    """
    This class defines the abstract class for a tensorflow model for the primitives.
    """

    def __init__(self, 
                 statedim, 
                 actiondim, 
                 k,
                 hidden_layer=64,
                 variance=10000):

        self.hidden_layer = hidden_layer
        self.variance = variance
        
        super(JHUJigSawsModel, self).__init__(statedim, actiondim, k)


    def createPolicyNetwork(self):

        return continuousTwoLayerReLU(self.statedim[0],
                                      self.actiondim[1],
                                      self.variance,
                                      self.hidden_layer)  


    def createTransitionNetwork(self):

        return multiLayerPerceptron(self.statedim[0],
                                    2,
                                    self.variance,
                                    self.hidden_layer)

