
from .TabularModel import TabularModel
import numpy as np
import scipy.special
from sklearn.preprocessing import normalize

"""
Defines a linear logistic model
"""


class LimitedTabularModel(TabularModel):

    def __init__(self,statedim, actiondim, unnormalized=False, limit=10):
      
        self.theta_map = {}
        self.isTabular = True
        self.actionbias = np.random.choice(np.arange(0,actiondim))
        self.limit = limit

        super(LimitedTabularModel, self).__init__(statedim, actiondim, unnormalized)


    def descent(self, grad_theta, learning_rate):

        super(LimitedTabularModel, self).descent(grad_theta, learning_rate)

        #project onto top 10
        most_important = [(self.theta_map[k],k) for k in self.theta_map]
        most_important.sort(reverse=True)
        for t in most_important[min(self.limit, len(most_important)):-1]:
            self.theta_map[t[1]] =  self.smoothing

