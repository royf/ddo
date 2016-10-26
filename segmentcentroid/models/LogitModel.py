
from .AbstractModel import AbstractModel
import numpy as np
import scipy.special

"""
Defines a linear logistic model
"""


class LogitModel(AbstractModel):

    def __init__(self,statedim, actiondim, unnormalized=False):
      
        self.theta = 10*np.random.rand(statedim, actiondim)

        super(LogitModel, self).__init__(statedim, actiondim, discrete=True, unnormalized=unnormalized)

    #returns a probability distribution over actions
    def eval(self, s):
        linproj = np.dot(self.theta.T, s.T)
        result = []
        for i in range(0,linproj.shape[0]):
            result.append(scipy.special.expit(linproj[i]))

        if self.unnormalized:
            return np.squeeze(np.array(result))
        else:
            return np.squeeze(np.array(result)/np.sum(np.array(result)))

    #return the log derivative log \nabla_\theta \pi(s)
    def log_deriv(self, s, a):
        linproj = np.dot(self.theta.T, s.T)
        gradient = np.zeros((self.statedim, self.actiondim))

        for i in range(0, self.actiondim):
            
            if a == i:
                gradient[:,i] = s - linproj[i]
            else:
                gradient[:,i] = - linproj[i]

        return gradient

    def descent(self, grad_theta, learning_rate):
        self.theta = self.theta + learning_rate*grad_theta
