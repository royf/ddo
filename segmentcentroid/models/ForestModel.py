
from .AbstractModel import AbstractModel
import numpy as np
import scipy.special

from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import NotFittedError
"""
Defines a linear logistic model
"""


class ForestModel(AbstractModel):

    def __init__(self,statedim, actiondim, unnormalized=False):
      
        #self.theta = 10*np.random.rand(statedim, actiondim)
        self.rf = RandomForestClassifier()
        self.isTabular = True

        super(ForestModel, self).__init__(statedim, actiondim, discrete=True, unnormalized=unnormalized)

    #returns a probability distribution over actions
    def eval(self, s):
        if self.actiondim == 1:
            try:
                self.rf.predict_proba(s)
                return self.rf.predict_proba(s)[0,1]
            except NotFittedError:
                return np.random.rand()

        try:
            a = self.rf.predict_proba(s)
            return np.ravel(a)
        except NotFittedError:
            rv = np.random.rand(4)
            return rv/np.sum(rv)


    #return the log derivative log \nabla_\theta \pi(s)
    def log_deriv(self, s, a):
        return (np.ravel(s),a)

    def descent(self, grad_theta, learning_rate):
        
        #self.rf = DecisionTreeClassifier(max_depth=3, random_state=1)

        #if self.actiondim == 1:
            #print([g[1][1] for g in grad_theta])

        X = []
        W = []
        Y = []

        for obs in grad_theta:
            W.append(obs[0])
            X.append(obs[1][0])
            Y.append(obs[1][1])

        self.rf.fit(X,Y,sample_weight=W)



