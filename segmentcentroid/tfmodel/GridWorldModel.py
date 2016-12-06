from .TFModel import TFNetworkModel
import tensorflow as tf
import numpy as np

class GridWorldModel(TFModel):
    
    """
    This class defines the abstract class for a tensorflow model for the primitives.
    """

    def __init__(self, 
                 statedim, 
                 actiondim, 
                 k,
                 hidden_layer=32):

        self.hidden_layer = hidden_layer
        self.init_var = init_var
        self.init_scale = init_scale
        
        super(GridWorldSoftMaxModel, self).__init__(statedim, actiondim, k)


    def createPolicyNetwork(self):

        return multiLayerPerceptron(self.statedim[0], 
                                    self.actiondim[0],
                                    self.hidden_layer)

    def createTransitionNetwork(self):

        return multiLayerPerceptron(self.statedim[0], 
                                    2,
                                    self.hidden_layer)

        

