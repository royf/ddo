
import numpy as np
import copy

import theano
import theano.tensor as T
from theano import function

"""
This class defines the main inference logic
"""

class SegCentroidInferenceDiscrete(object):

    def __init__(self, policy_class, k):
        self.policy_class = policy_class
        self.k = k


    #X is a list of segmented trajectories
    def fit(self, X, statedims, actionsize, max_iters=1, learning_rate=0.01):
        
        #create k initial policies
        x = T.matrix('x') 
        policies = [copy.copy(self.policy_class(x, statedims* 1, actionsize)) for i in range(0,self.k)]


        #initialize q and P for iterations
        q = np.matrix(np.ones((len(X),self.k)))/self.k
        P = np.matrix(np.ones((self.k,1)))/self.k

        m = len(X)


        #Outer Loop For Gradient Descent
        for it in range(0, max_iters):

            q, P = self._updateQP(X, policies, q, P)

            y = T.ivector('y')

            for seg in range(0,self.k):

                for plan in range(0, m):

                    batch = 0

                    #to fix here...
                    for t in range(0, len(X[plan])):

                        cost = policies[seg].log_likelihood(y)

                        action = X[plan][t][1]

                        lvector = np.squeeze(np.zeros((self.k,1)))
                        lvector[action] = 1 - 1e-6

                        obs = np.matrix(X[plan][t][0])

                        g_W = function( ,T.grad(cost=cost, wrt=policies[seg].W), allow_input_downcast=True)
                        #print obs.shape, lvector.shape
                        theano.function( inputs=[x,y], outputs=cost, updates=updates)
                        policies[seg].W = policies[seg].W + learning_rate*g_W(obs, lvector)



    """
    Defines the inner loops
    """

    def _updateQP(self, X, policies, q, P):

        #how many trajectories are there in X
        m = len(X)

        #for each trajectory
        for plan in range(0, m):

            #for each segments
            for seg in range(0, self.k):

                q[plan, seg] = P[seg]*self._segLikelihood(X[plan], policies[seg])
      
        normalization = np.matrix(np.sum(q, axis=1))
        normalization_matrix = np.tile(1/normalization, [1,self.k])
        q = np.multiply(q, normalization_matrix)
        P = np.matrix(np.sum(q, axis=0)).T/m
            
        return q,P



    def _segLikelihood(self, traj, policy):
        product = 1

        for t in range(0, len(traj)):

            obs = np.matrix(traj[t][0])

            pred = np.squeeze(policy.eval(obs))

            action = traj[t][1]
            
            preda = pred[action]

            product = preda * product

        return product


            










