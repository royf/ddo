import tensorflow as tf
import numpy as np
from segmentcentroid.inference.forwardbackward import ForwardBackward
from tensorflow.python.client import timeline

class TFModel(object):
    """
    This class defines the basic data structure for a hierarchical control model. This
    is a wrapper that handles I/O, Checkpointing, and Training Primitives.
    """

    def __init__(self, 
                 statedim, 
                 actiondim, 
                 k,
                 checkpoint_file='/tmp/model.bin',
                 checkpoint_freq=10):
        """
        Create a TF model from the parameters

        Positional arguments:
        statedim -- numpy.ndarray defining the shape of the state-space
        actiondim -- numpy.ndarray defining the shape of the action-space
        k -- float defining the number of primitives to learn

        Keyword arguments:
        checkpoint_file -- string filname to store the learned model
        checkpoint_freq -- int iter % checkpoint_freq the learned model is checkpointed
        """

        self.statedim = statedim 
        self.actiondim = actiondim 

        self.k = k

        self.sess = tf.Session()
        self.initialized = False

        self.checkpoint_file = checkpoint_file
        self.checkpoint_freq = checkpoint_freq

        self.initialize()
        self.fb = ForwardBackward(self)

        self.saver = tf.train.Saver()


    def initialize(self):
        """
        The initialize command is implmented by all subclasses and designed 
        to initialize whatever internal state is needed.
        """

        raise NotImplemented("Must implement an initialize function")


    def restore(self):
        """
        Restores the model from the checkpointed file
        """

        self.saver.restore(self.sess, self.checkpoint_file)

    def save(self):
        """
        Saves the model to the checkpointed file
        """

        self.saver.save(self.sess, self.checkpoint_file)

    def evalpi(self, index, traj):
        """
        Returns the probability of action a at state s for primitive index i

        Positional arguments:
        index -- int index of the required primitive in {0,...,k}
        traj -- a trajectory

        Returns:
        float -- probability
        """

        if index >= self.k:
            raise ValueError("Primitive index is greater than the number of primitives")

        X, A = self.formatTrajectory(traj)

        return self._evalpi(index, X, A)


    def evalpsi(self, index, traj):
        """
        Returns the probability of action a at state s

        Positional arguments:
        index -- int index of the required primitive in {0,...,k}
        traj -- a trajectory

        Returns:
        float -- probability
        """

        if index >= self.k:
            raise ValueError("Primitive index is greater than the number of primitives")


        X, _ = self.formatTrajectory(traj)

        return self._evalpsi(index, X)


    def _evalpi(self, index, X, A):
        """
        Sub classes must implment this actual execution routine to eval the probability

        Returns:
        float -- probability
        """
        raise NotImplemented("Must implement an _evalpi function")


    def _evalpsi(self, index, X):
        """
        Sub classes must implment this actual execution routine to eval the probability

        Returns:
        float -- probability
        """
        raise NotImplemented("Must implement an _evalpsi function")

    def getLossFunction(self):
        """
        Sub classes must implement a function that returns the loss and trainable variables

        Returns:
        loss -- tensorflow function
        pivars -- variables that handle policies
        psivars -- variables that handle transitions
        """
        raise NotImplemented("Must implement a getLossFunction")


    def dataTransformer(self, trajectory):
        """
        Sub classes can implement a data augmentation class. The default is the identity transform

        Positional arguments: 
        trajectory -- input is a single trajectory

        Returns:
        trajectory
        """
        return trajectory


    """
    ####
    Fitting functions. Below we include functions for fitting the models.
    These are mostly for convenience
    ####
    """

    def sampleBatch(self, X):
        """
        sampleBatch executes the forward backward algorithm and returns
        a single batch of data to train on.

        Positional arguments:
        X -- a list of trajectories. Each trajectory is a list of tuples of states and actions
        dataTransformer -- a data augmentation routine
        """

        #loss, pivars, psivars = self.getLossFunction()

        traj_index = np.random.choice(len(X))

        trajectory = self.dataTransformer(X[traj_index])

        weights = self.fb.fit([trajectory])

        feed_dict = {}
        Xm, Am = self.formatTrajectory(trajectory)

        #print(Xm.shape, Am.shape, weights[0][0][:,0].shape, weights[0][1][:,0].shape)

        print("##Q##",weights[0][0])

        for j in range(self.k):
            feed_dict[self.pivars[j][0]] = Xm[1:len(X[traj_index])-1,:]
            feed_dict[self.pivars[j][1]] = Am[1:len(X[traj_index])-1,:]
            feed_dict[self.pivars[j][2]] = np.reshape(weights[0][0][:,j], (Xm.shape[0],1))[1:len(X[traj_index])-1,:]

            feed_dict[self.psivars[j][0]] = Xm[1:len(X[traj_index])-1,:]
            feed_dict[self.psivars[j][1]] = self.formatTransitions(weights[0][1][:,j])[1:len(X[traj_index])-1,:]
            feed_dict[self.psivars[j][2]] = np.reshape(weights[0][1][:,j], (Xm.shape[0],1))[1:len(X[traj_index])-1,:]

        return feed_dict
        
    def formatTrajectory(self, 
                         trajectory, 
                         statedim=None, 
                         actiondim=None):
        """
        Internal method that unzips a trajectory into a state and action tuple

        Positional arguments:
        trajectory -- a list of state and action tuples.
        """

        #print("###", statedim, actiondim)

        if statedim == None:
            statedim = self.statedim

        if actiondim == None:
            actiondim = self.actiondim

        sarraydims = [s for s in statedim]
        sarraydims.insert(0, len(trajectory))
        #creates an n+1 d array 

        aarraydims = [a for a in actiondim]
        aarraydims.insert(0, len(trajectory))
        #creates an n+1 d array 

        X = np.zeros(tuple(sarraydims))
        A = np.zeros(tuple(aarraydims))

        for t in range(len(trajectory)):
            #print(t, trajectory[t][0], trajectory[t][0].shape, statedim)
            s = np.reshape(trajectory[t][0], statedim)
            a = np.reshape(trajectory[t][1], actiondim)

            X[t,:] = s
            A[t,:] = a


        #special case for 2d arrays
        if len(statedim) == 2 and \
           statedim[1] == 1:
           X = np.squeeze(X,axis=2)
           #print(X.shape)
        
        if len(actiondim) == 2 and \
           actiondim[1] == 1:
           A = np.squeeze(A,axis=2)
           #print(A.shape)

        return X, A

    def formatTransitions(self, transitions):
        """
        Internal method that turns a transition sequence (array of floats [0,1])
        into an encoded array [1-a, a]
        """

        X = np.zeros((len(transitions),2))
        for t in range(len(transitions)-1):
            X[t,0] = 1- transitions[t]
            X[t,1] = transitions[t]
        
        return X


    def getOptimizationVariables(self, opt):
        """
        This is an internal method that returns the tensorflow refs
        needed for optimization.

        Positional arguments:
        opt -- a tf.optimizer
        """
        loss, pivars, psivars = self.getLossFunction()
        train = opt.minimize(loss)
        init = tf.initialize_all_variables()
        return (loss, train, init, pivars, psivars)


    def startTraining(self, opt):
        """
        This method initializes the training routine

        opt -- is the chosen optimizer to use
        """

        self.loss, self.train, self.init, self.pivars, self.psivars = self.getOptimizationVariables(opt)
        self.sess.run(self.init)
        self.initialized = True
        tf.get_default_graph().finalize()


    def train(self, opt, X, iterations, subiterations):
        """
        This method trains the model on a dataset weighted by the forward 
        backward algorithm

        Positional arguments:
        opt -- a tf.optimizer
        X -- a list of trajectories
        iterations -- the number of iterations
        subiterations -- the number of iterations per forward-backward algorithm
        """

        if not self.initialized:
            self.startTraining(opt)
            

        for it in range(iterations):

            if it % self.checkpoint_freq == 0:
                print("Checkpointing Train", it, self.checkpoint_file)
                self.save()

            batch = self.sampleBatch(X)

            print("Iteration", it, np.argmax(self.fb.Q, axis=1))
            
            for i in range(subiterations):
                self.sess.run(self.train, batch)




        
