import tensorflow as tf
import numpy as np
from segmentcentroid.inference.forwardbackward import ForwardBackward

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
        self.trained = False

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

    def evalpi(self, index, s, a):
        """
        Returns the probability of action a at state s for primitive index i

        Positional arguments:
        index -- int index of the required primitive in {0,...,k}
        s -- state an np.ndarray in the proper shape
        a -- action an np.ndarray in the proper shape

        Returns:
        float -- probability
        """

        if index >= self.k:
            raise ValueError("Primitive index is greater than the number of primitives")

        s = np.reshape(s, self.statedim)

        a = np.reshape(a, self.actiondim)

        return self._evalpi(index, s, a)


    def evalpsi(self, index, s):
        """
        Returns the probability of action a at state s

        Positional arguments:
        index -- int index of the required primitive in {0,...,k}
        s -- state an np.ndarray in the proper shape

        Returns:
        float -- probability
        """

        if index >= self.k:
            raise ValueError("Primitive index is greater than the number of primitives")

        s = np.reshape(s, self.statedim)

        return self._evalpsi(index, s)


    def _evalpi(self, index, s, a):
        """
        Sub classes must implment this actual execution routine to eval the probability

        Returns:
        float -- probability
        """
        raise NotImplemented("Must implement an _evalpi function")


    def _evalpsi(self, index, s):
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
        """

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

    def samplePretrainBatch(self, X):
        """
        samplePretainBatch executes returns a batch of data with random weights (no forward backward)

        Positional arguments:
        X -- a list of trajectories. Each trajectory is a list of tuples of states and actions
        """

        loss, pivars, psivars = self.getLossFunction()

        traj_index = np.random.choice(len(X))
        weights = self.fb.randomWeights([X[traj_index]])
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

        
    def formatTrajectory(self, trajectory):
        """
        Internal method that unzips a trajectory into a state and action tuple

        Positional arguments:
        trajectory -- a list of state and action tuples.
        """

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
        loss = self.getLossFunction()[0]
        train = opt.minimize(loss)
        init = tf.initialize_all_variables()
        return (loss, train, init)


    def pretrain(self, opt, X, iterations):
        """
        This method pre-trains the model on a randomly weighted dataset

        Positional arguments:
        opt -- a tf.optimizer
        X -- a list of trajectories
        iterations -- the number of iterations
        """
        loss, train, init = self.getOptimizationVariables(opt)
        
        self.sess.run(init)
        self.trained = True

        for it in range(iterations):

            if it % 100 == 0:
                print("Pretrain Iteration=", it)

            batch = self.samplePretrainBatch(X)
            self.sess.run(train, batch)


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

        loss, train, init = self.getOptimizationVariables(opt)

        if not self.trained:
            self.sess.run(init)

        for it in range(iterations):

            if it % self.checkpoint_freq == 0:
                print("Checkpointing Train", it, self.checkpoint_file)
                self.save()

            batch = self.sampleBatch(X)

            print("Iteration", it, np.argmax(self.fb.Q, axis=1))
            
            for i in range(subiterations):
                self.sess.run(train, batch)




class TFNetworkModel(TFModel):
    """
    This class defines a common instantiation of the abstract class TFModel
    where all of the policies and transitions are of an identical type.
    """

    def __init__(self, 
                 statedim, 
                 actiondim, 
                 k):
        """
        Create a model from the parameters

        Positional arguments:
        statedim -- numpy.ndarray defining the shape of the state-space
        actiondim -- numpy.ndarray defining the shape of the action-space
        k -- float defining the number of primitives to learn
        """

        self.policy_networks = []
        self.transition_networks = []

        super(TFNetworkModel, self).__init__(statedim, actiondim, k)


    def getLossFunction(self):

        """
        Returns a loss function that sums over policies and transitions
        """

        loss_array = []

        pi_vars = []
        for i in range(0, self.k):
            loss_array.append(self.policy_networks[i]['wlprob'])
            pi_vars.append((self.policy_networks[i]['state'], 
                            self.policy_networks[i]['action'], 
                            self.policy_networks[i]['weight']))

        psi_vars = []
        for i in range(0, self.k):
            loss_array.append(self.transition_networks[i]['wlprob'])
            psi_vars.append((self.transition_networks[i]['state'], 
                            self.transition_networks[i]['action'], 
                            self.transition_networks[i]['weight']))

        return tf.reduce_sum(loss_array), pi_vars, psi_vars



    def initialize(self):
        """
        Initializes the internal state
        """

        for i in range(0, self.k):
            self.policy_networks.append(self.createPolicyNetwork())

        for i in range(0, self.k):
            self.transition_networks.append(self.createTransitionNetwork())


    #returns a probability distribution over actions
    def _evalpi(self, index, s, a):
        feed_dict = {self.policy_networks[index]['state']: s.reshape((1, self.statedim[0])),
                     self.policy_networks[index]['action']: a.reshape((1,self.actiondim[0]))}

        dist = np.ravel(self.sess.run(self.policy_networks[index]['prob'], feed_dict))

        if self.policy_networks[index]['discrete']:
            encoded_action = np.argwhere(a > 0)[0][0]
            return dist[encoded_action]/np.sum(dist)
        else: 
            return dist
            

    #returns a probability distribution over actions
    def _evalpsi(self, index, s):
        feed_dict = {self.transition_networks[index]['state']: s.reshape((1, self.statedim[0]))}
        encoded_action = 1
        dist = np.ravel(self.sess.run(self.transition_networks[index]['prob'], feed_dict))

        if not self.transition_networks[index]['discrete']:
            raise ValueError("Transition function must be discrete")

        return dist[encoded_action]/np.sum(dist)




        
