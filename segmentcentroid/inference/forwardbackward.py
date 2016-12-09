
import numpy as np
import copy
from sklearn.preprocessing import normalize
from scipy.misc import logsumexp
import datetime

class ForwardBackward(object):
    """
    The ForwardBackward class performs one forward and backward pass
    of the algorithm returning the tables Q[t,h], B[t,h]. These define
    the weighting functions for the gradient step. The class is initialized
    with model and logging parameters, and is fit with a list of trajectories.
    """

    def __init__(self, model, verbose=False):
        """
        This initializes the FB algorithm with a TFmodel

        Positional arguments:

        model -- TFModel This is a model object which is a wrapper for a tensorflow model
        verbose -- Boolean True means that FB algorithm will print logging output to stdout
        """

        self.verbose = verbose

        if self.verbose:
            print("[HC: Forward-Backward] model=",model)

        self.model = model
        self.k = model.k

        self.X = None
        self.Q = None
        self.B = None

        self.P = np.ones((self.k, self.k))/self.k

        self.Pbar = np.ones((self.k, self.k), dtype='float128')/2


    def fit(self, trajectoryList):
        """
        Each trajectory is a sequence of tuple (s,a) where both s and a are numpy
        arrays. 

        Positional arguments:

        trajectoryList -- is a list of trajectories.

        Returns:
        A dict of trajectory id which mapt to the the weights Q, B, P
        """

        iter_state = {}

        if self.verbose:
            print("[HC: Forward-Backward] found ",len(trajectoryList),"trajectories")

        for i, traj in enumerate(trajectoryList):

            if self.verbose:
                print("[HC: Forward-Backward] fitting ", i, len(traj))
            
            start = datetime.datetime.now()

            self.init_iter(i, traj)

            print("Init Iter", datetime.datetime.now()-start)
            
            iter_state[i] = self.fitTraj(traj)

        return iter_state


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

        #todo fix batch it up
        if tabulate:
            for t in range(0,len(X)-1):
                for h in range(0, self.k):
                    state = X[t][0]
                    next_state = X[t+1][0]
                    action = X[t][1]
                    self.pi[t,h] = self._pi_a_giv_s(state, action, h)
                    self.psi[t,h] = self._pi_term_giv_s(next_state,h)

        


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

        start = datetime.datetime.now()

        self.forward()

        print("Forward", datetime.datetime.now()-start)

        start = datetime.datetime.now()

        self.backward()

        print("Backward", datetime.datetime.now()-start)

        
        Qunorm = np.add(self.fq,self.bq)

        self.Qnorm = logsumexp(Qunorm, axis=1)

        self.Q = np.exp(Qunorm - self.Qnorm[:, None])

        #print(self.Qnorm)
        
        #print("[HC: Forward-Backward] Q Update", np.argmax(self.Q, axis=1), len(np.argwhere(np.argmax(self.Q, axis=1) > 0))) 

        self.updateTransitionProbability()      

        for t in range(len(self.X)):
            update = np.exp(self.termination(t) - self.Qnorm[t])
            #print(update)
            self.B[t,:] = update

        if self.verbose:
            print("[HC: Forward-Backward] B Update", np.argmax(self.B, axis=1)) 


        return self.Q[0:len(X),:], self.B[0:len(X),:], self.P


    def randomWeights(self, X):
        """
        This returns a dummy object back with random weights used for pretraining

        Positional arguments:

        X -- is a list of s,a tuples.
        
        """

        self.init_iter(0, X[0], tabulate=False)

        self.X = X[0]

        self.Q = np.random.rand(len(self.X),self.k)
  

        for t in range(len(self.X)):
            self.B[t,:] = np.ones((1,self.k))


        return self.Q[0:len(self.X),:], self.B[0:len(self.X),:]
                


    def forward(self):
        """
        Performs a foward pass, updates the state
        """

        t = len(self.X)-1

        #initialize table
        forward_dict = {}
        for h in range(self.k):
            forward_dict[(0,h)] = 0

        #dynamic program
        for cur_time in range(t):

            for hp in range(self.k):

                forward_dict[(cur_time+1, hp)] = \
                            logsumexp([ forward_dict[(cur_time, h)] + \
                                        np.log(self.pi[cur_time,h]) + \
                                        np.log(self.P[hp,h]) + 
                                        np.log(self.psi[cur_time,h] + \
                                              (1-self.psi[cur_time,h])*(hp == h))\
                                        for h in range(self.k)
                                     ])

                #if self.verbose:
                #    print("[HC: Forward-Backward] Forward DP Update", forward_dict[(cur_time+1, hp)], hp, cur_time+1, [ np.log(self._pi_a_giv_s(state,action,h)) for h in range(self.k)]) 

        for k in forward_dict:
            self.fq[k[0],k[1]] = forward_dict[k]


    def backward(self):
        """
        Performs a backward pass, updates the state
        """

        t = 0

        #initialize table
        backward_dict = {}
        for h in range(self.k):
            backward_dict[(len(self.X),h)] = 0

        rt = np.arange(len(self.X), t, -1)

        #dynamic program
        for cur_time in rt:

            for h in range(self.k):

                clipped_time = min(cur_time,len(self.X)-1)
                
                backward_dict[(cur_time-1, h)] = \
                    logsumexp([ backward_dict[(cur_time, hp)] +\
                                np.log(self.pi[clipped_time,h]) +\
                                np.log((self.P[hp,h]*self.psi[clipped_time,h] + \
                                                                 (1-self.psi[clipped_time,h])*(hp == h)
                                                                ))\
                                for hp in range(self.k)
                              ])

                if self.verbose:
                    print("[HC: Forward-Backward] Backward DP Update", backward_dict[(cur_time-1, h)], h, cur_time-1) 

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

            if self.verbose:
                    print("[HC: Forward-Backward] Termination Update", termination[h], h) 


        return [termination[h] for h in range(self.k)]


    def updateTransitionProbability(self):
        """
        Updates the transition probabilities P
        """

        for t in range(len(self.X)-1):
            qt = self.Q[t,:]
            qtp = self.Q[t+1,:]
            for i in range(self.k):
                for j in range(self.k):
                    self.Pbar[i,j] = self.Pbar[i,j] + np.exp(np.logaddexp(qt[i], qtp[j]))

        self.P = normalize(self.Pbar, norm='l1', axis=0)


    def _pi_a_giv_s(self, s, a, index):
        """
        Wrapper for the model function
        """
        return self.model.evalpi(index, s, a)


    def _pi_term_giv_s(self, s, index):
        """
        Wrapper for the model function
        """
        return self.model.evalpsi(index, s)




        
