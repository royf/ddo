import tensorflow as tf
import numpy as np
from segmentcentroid.inference.forwardbackward import ForwardBackward

class TFModel(object):
    
    """
    This class defines the abstract class for a tensorflow model for the primitives.
    """

    def __init__(self, 
                 statedim, 
                 actiondim, 
                 k):
        """
        Defines the state-space and action-space and number of primitves
        """

        self.statedim = statedim #numpy shape
        self.actiondim = actiondim #numpy shape
        self.k = k
        self.fb = None
        self.sess = tf.Session()

        self.initialize()
        self.fb = ForwardBackward(self)


    def initialize(self):
        raise NotImplemented("Must implement an initialize function")

    def evalpi(self, index, s, a):
        """
        Returns the probability of action a at state s
        """

        if index >= self.k:
            raise ValueError("Primitive index is greater than the number of primitives")

        s = np.reshape(s, self.statedim)

        a = np.reshape(a, self.actiondim)

        return self._evalpi(index, s, a)

    #returns a probability distribution over actions
    def _evalpi(self, index, s, a):
        raise NotImplemented("Must implement an _evalpi function")


    def evalpsi(self, index, s):
        """
        Returns the probability of action a at state s
        """

        if index >= self.k:
            raise ValueError("Primitive index is greater than the number of primitives")

        s = np.reshape(s, self.statedim)

        return self._evalpsi(index, s)

    #returns a probability distribution over actions
    def _evalpsi(self, index, s):
        raise NotImplemented("Must implement an _evalpsi function")


    #returns a tensfor flow loss function, with a dict of all the training 
    #variables
    def getLossFunction(self):
        raise NotImplemented("Must implement a getLossFunction")


    """
    Fitting primitives
    """

    #samples one stochastic gradient batch
    def sampleBatch(self, X):
        loss, pivars, psivars = self.getLossFunction()
        traj_index = np.random.choice(len(X))
        weights = self.fb.fit([X[traj_index]])
        feed_dict = {}
        Xm, Am = self.formatTrajectory(X[traj_index])

        for j in range(self.k):
            feed_dict[pivars[j][0]] = Xm[1:len(X[traj_index])-1,:]
            feed_dict[pivars[j][1]] = Am[1:len(X[traj_index])-1,:]
            feed_dict[pivars[j][2]] = np.reshape(weights[0][0][:,j], (Xm.shape[0],1))[1:len(X[traj_index])-1,:]

            feed_dict[psivars[j][0]] = Xm[1:len(X[traj_index])-1,:]
            feed_dict[psivars[j][1]] = self.formatTransitions(weights[0][1][:,j])[1:len(X[traj_index])-1,:]
            feed_dict[psivars[j][2]] = np.reshape(weights[0][1][:,j], (Xm.shape[0],1))[1:len(X[traj_index])-1,:]

        return feed_dict

    #samples one stochastic gradient batch
    def samplePretrainBatch(self, X):
        loss, pivars, psivars = self.getLossFunction()
        traj_index = np.random.choice(len(X))
        weights = self.fb.randomWeights([X[traj_index]])
        #print(weights[0][1])
        feed_dict = {}
        Xm, Am = self.formatTrajectory(X[traj_index])

        for j in range(self.k):
            feed_dict[pivars[j][0]] = Xm[1:len(X[traj_index])-1,:]
            feed_dict[pivars[j][1]] = Am[1:len(X[traj_index])-1,:]
            feed_dict[pivars[j][2]] = np.reshape(weights[0][:,j], (Xm.shape[0],1))[1:len(X[traj_index])-1,:]

            feed_dict[psivars[j][0]] = Xm[1:len(X[traj_index])-1,:]
            feed_dict[psivars[j][1]] = self.formatTransitions(weights[1][:,j])[1:len(X[traj_index])-1,:]
            feed_dict[psivars[j][2]] = np.reshape(weights[1][:,j], (Xm.shape[0],1))[1:len(X[traj_index])-1,:]

        return feed_dict

        
    #helper method that formats the trajectory
    def formatTrajectory(self, trajectory):

        if self.statedim[1] != 1:
            raise NotImplemented("Currently doesn't support more complex trajectories")

        if self.actiondim[1] != 1:
            raise NotImplemented("Currently doesn't support more complex trajectories")

        sdim = self.statedim[0]
        adim = self.actiondim[0]

        X = np.zeros((len(trajectory),sdim))
        A = np.zeros((len(trajectory),adim))

        for t in range(len(trajectory)):
            s = np.transpose(np.reshape(trajectory[t][0], self.statedim))
            a = np.transpose(np.reshape(trajectory[t][1], self.actiondim))

            X[t,:] = s
            A[t,:] = a

        return X,A

    #helper method that formats the transitions
    def formatTransitions(self, transitions):
        X = np.zeros((len(transitions),2))
        for t in range(len(transitions)-1):
            X[t,0] = 1- transitions[t]
            X[t,1] = transitions[t]
        
        return X





        
