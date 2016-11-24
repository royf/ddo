
from .AbstractModel import AbstractModel
import numpy as np
import scipy.special

"""
Defines a linear logistic model
"""


class LogitModel(AbstractModel):

    def __init__(self,statedim, actiondim, unnormalized=False):
      
        self.theta = 10*np.random.randn(statedim, actiondim)
        self.isTabular = False
        self.smoothing = 0.02

        super(LogitModel, self).__init__(statedim, actiondim, discrete=True, unnormalized=unnormalized)

    #returns a probability distribution over actions
    def eval(self, s):
        linproj = np.dot(self.theta.T, s.T)
        result = []
        for i in range(0,linproj.shape[0]):
            exp = np.ravel(scipy.special.expit(linproj[i]))
            result.append(max(exp, self.smoothing))

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


class BinaryLogitModel(AbstractModel):

    def __init__(self,statedim, actiondim, unnormalized=False):
      
        self.theta = 10*np.random.randn(statedim, 1)
        self.isTabular = False
        self.smoothing = 0.02

        super(BinaryLogitModel, self).__init__(statedim, actiondim, discrete=True, unnormalized=unnormalized)

    #returns a probability distribution over actions
    def eval(self, s):
        linproj = np.dot(self.theta.T, s.T)

        return max(np.squeeze(scipy.special.expit(linproj)), self.smoothing)

    #return the log derivative log \nabla_\theta \pi(s)
    def log_deriv(self, s, a):
        linproj = np.dot(self.theta.T, s.T)
        gradient = np.zeros((self.statedim, 1))
            
        if a == 0:
            gradient[:,0] = s - linproj
        else:
            gradient[:,0] = - linproj

        return gradient

    def descent(self, grad_theta, learning_rate):
        self.theta = self.theta + learning_rate*grad_theta

