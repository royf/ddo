
from AbstractModel import AbstractModel
import numpy as np
import scipy.special

"""
Defines a linear logistic model
"""


class LogitModel(AbstractModel):

    def __init__(self,statedim, actiondim):
      
        self.theta = np.random.randn(statedim, actiondim)

        super(LogitModel, self).__init__(statedim, actiondim, discrete=True)

    #returns a probability distribution over actions
    def eval(self, s):
        linproj = np.dot(self.theta.T, s.T)
        result = []
        for i in range(0,linproj.shape[0]):
            result.append(scipy.special.expit(linproj[i]))

        return np.squeeze(np.array(result)/np.sum(np.array(result)))

    #return the log derivative log \nabla_\theta \pi(s)
    def log_deriv(self, s, a):
        
        self.eval(s)
