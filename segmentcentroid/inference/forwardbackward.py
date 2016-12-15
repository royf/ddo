
import numpy as np
import copy
from scipy.misc import logsumexp


class ForwardBackward(object):
    """
    The ForwardBackward class performs one forward and backward pass
    of the algorithm returning the tables Q[t,h], B[t,h]. These define
    the weighting functions for the gradient step. The class is initialized
    with model and logging parameters, and is fit with a list of trajectories.
    """

    def __init__(self, model):
        """
        This initializes the FB algorithm with a TFmodel

        Positional arguments:

        model -- TFModel This is a model object which is a wrapper for a tensorflow model
        """

        self.model = model
        self.k = model.k

        self.X = None
        self.Q = None
        self.B = None

        self.P = np.ones((self.k, self.k))/self.k



    def fit(self, trajectoryList):
        """
        Each trajectory is a sequence of tuple (s,a) where both s and a are numpy
        arrays. 

        Positional arguments:

        trajectoryList -- is a list of trajectories.

        Returns:
        A dict of trajectory id which mapt to the the weights Q, B
        """

        iter_state = {}

        for i, traj in enumerate(trajectoryList):

            if not self.isValid(traj):
                raise ValueError("The provided trajectory does not match the dimensions of the model")

            self.init_iter(i, traj)
            
            iter_state[i] = self.fitTraj(traj)

        return iter_state


    def isValid(self, traj):
        """
        Validates that the trajectory matches the state and action dimensions of the model.

        Positional argument:

        traj -- a list of tuples t[0] is a state, and t[1] is an action

        Returns:
        Boolean
        """

        for t in traj:
            if t[0].shape != self.model.statedim or \
               t[1].shape != self.model.actiondim:
               return False

        return True


    def init_iter(self, index, X, tabulate=True):
        """
        Internal method that initializes the state variables

        Positional arguments:
        index -- int trajectory id
        X -- trajectory
        """

        self.Q = np.ones((len(X)+1, self.k) , dtype='float128')/self.k
        self.fq = np.ones((len(X)+1, self.k) , dtype='float128')/self.k
        self.bq = np.ones((len(X)+1, self.k), dtype='float128')/self.k
        self.B = np.ones((len(X)+1, self.k), dtype='float128')/2

        self.pi = np.ones((len(X), self.k))
        self.psi = np.ones((len(X), self.k))

        if tabulate:
            for h in range(0, self.k):
                self.pi[:,h] = self.model.evalpi(h,X)
                self.psi[:,h] = self.model.evalpsi(h,X)

        
    def fitTraj(self, X):
        """
        This function runs one pass of the Forward Backward algorithm over
        a trajectory.

        Positional arguments:

        X -- is a list of s,a tuples.

        Return:
        Two tables of weights Q, B which define the weights for the gradient step
        """

        self.X = X

        self.forward()

        self.backward()
        
        Qunorm = np.add(self.fq,self.bq)

        self.Qnorm = logsumexp(Qunorm, axis=1)

        self.Q = np.exp(Qunorm - self.Qnorm[:, None])  

        for t in range(len(self.X)):
            update = np.exp(self.termination(t) - self.Qnorm[t])

            #set the last/first time point to 1
            if t != len(self.X) - 1 or t != 0:
                self.B[t,:] = update
            else:
                self.B[t,:] = np.ones((1, self.k))

        return self.Q[0:len(X),:], self.B[0:len(X),:], self.P


    def forward(self):
        """
        Performs a foward pass, updates the 
        """

        #initialize table
        forward_dict = {}
        for h in range(self.k):
            forward_dict[(0,h)] = 0

        #dynamic program
        for cur_time in range(len(self.X)):

            for hp in range(self.k):

                forward_dict[(cur_time+1, hp)] = \
                            logsumexp([forward_dict[(cur_time, h)] + \
                                        np.log(self.pi[cur_time,h]) + \
                                        np.log(self.psi[cur_time,h]*self.P[hp,h] + \
                                              (1-self.psi[cur_time,h])*(hp == h))\
                                        for h in range(self.k)
                                     ])

        for k in forward_dict:
            self.fq[k[0],k[1]] = forward_dict[k]


    def backward(self):
        """
        Performs a backward pass, updates the state
        """

        #initialize table
        backward_dict = {}
        for h in range(self.k):
            backward_dict[(len(self.X)-1,h)] = 0

        rt = np.arange(len(self.X)-2, -1, -1)

        #dynamic program
        for cur_time in rt:

            for h in range(self.k):
                
                backward_dict[(cur_time, h)] = np.log(self.pi[cur_time,h]) + \
                    logsumexp([ backward_dict[(cur_time+1, hp)] +\
                                np.log((self.P[hp,h]*self.psi[cur_time,h] + \
                                        (1-self.psi[cur_time,h])*(hp == h)
                                       ))\
                                for hp in range(self.k)
                              ])

        for k in backward_dict:
            self.bq[k[0],k[1]] = backward_dict[k]


    def termination(self, t):
        """
        This function calculates B for a particular time step
        """

        state = self.X[t][0]
            
        if t+1 == len(self.X):
            next_state = state
        else:
            next_state = self.X[t+1][0]

        action = self.X[t][1]


        termination = {}

        for h in range(self.k):

            termination[h] = \
                logsumexp([self.fq[t,h] + \
                           np.log(self.pi[t,h]) + \
                           np.log(self.P[hp,h]) + \
                           np.log(self.psi[t,h]) + \
                           self.bq[t+1,hp] \
                           for hp in range(self.k)
                          ])

        return [termination[h] for h in range(self.k)]





        
