
from .AbstractModel import AbstractModel
import numpy as np
import scipy.special
from sklearn.preprocessing import normalize

"""
Defines a linear logistic model
"""


class TabularModel(AbstractModel):

    def __init__(self,statedim, actiondim, unnormalized=False):
      
        self.theta_map = {}
        self.isTabular = True
        #self.actionbias = np.random.choice(np.arange(0,actiondim))
        self.smoothing = 1e-6

        super(TabularModel, self).__init__(statedim, actiondim, discrete=True, unnormalized=unnormalized)

    #returns a probability distribution over actions
    def eval(self, s):
        if self.actiondim == 1:
            return self.evalBin(s)

        s = np.ravel(s)
        sp = self.state_to_tuple(s)

        result = np.zeros((self.actiondim,1))

        for i in range(0,self.actiondim):
            if (sp,i) not in self.theta_map:
                
                init = np.random.choice([self.smoothing, 1])

                result[i] = init
                self.theta_map[(sp,i)] = init
            else:
                result[i] = max(self.theta_map[(sp,i)],self.smoothing)

        #print(np.squeeze(result)/np.sum(np.array(result)))

        return np.squeeze(result)/np.sum(np.array(result))

    #returns a probability distribution over actions
    def evalBin(self, s):
        s = np.ravel(s)
        sp = self.state_to_tuple(s)

        result = np.zeros((2,1))

        for i in range(0,2):
            if (sp,i) not in self.theta_map:
                init = np.random.choice([self.smoothing, 1])
                result[i] = init
                self.theta_map[(sp,i)] = init
            else:
                result[i] = self.theta_map[(sp,i)]

        #print(np.squeeze(result)/np.sum(np.array(result)))

        return (np.squeeze(result)/np.sum(np.array(result)))[1]


    def log_deriv(self, s, a):
        s = np.ravel(s)
        sp = self.state_to_tuple(s)
        return (sp,a)

    def state_to_tuple(self, s):
        return tuple([i for i in s])

    def descent(self, grad_theta, learning_rate):

        #print(self.theta_map)

        for k in self.theta_map:
            self.theta_map[k] = max(self.smoothing, self.theta_map[k]-learning_rate)

        for obs in grad_theta:
            weight = obs[0]
            state = self.state_to_tuple(obs[1][0])
            action = obs[1][1]

            if (state,action) not in self.theta_map:
                self.theta_map[(state,action)] = max(weight, self.smoothing)

            self.theta_map[(state,action)] = max(weight,self.smoothing)

    def visited(self, s):
        return (self.state_to_tuple(s) in set([k[0] for k in self.theta_map]))
