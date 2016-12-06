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
                 k,
                 checkpoint_file='/tmp/model.bin',
                 checkpoint_freq=10):
        """
        Defines the state-space and action-space and number of primitves
        """

        self.statedim = statedim #numpy shape
        self.actiondim = actiondim #numpy shape
        self.k = k
        self.fb = None
        self.sess = tf.Session()
        self.trained = False

        self.checkpoint_file = checkpoint_file
        self.checkpoint_freq = checkpoint_freq

        self.initialize()
        self.fb = ForwardBackward(self)
        self.saver = tf.train.Saver()


    def initialize(self):
        raise NotImplemented("Must implement an initialize function")


    def restore(self):
        self.saver.restore(self.sess, self.checkpoint_file)

    def save(self):
        self.saver.save(self.sess, self.checkpoint_file)

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


    def getOptimizationVariables(self, opt):
        loss = self.getLossFunction()[0]
        train = opt.minimize(loss)
        init = tf.initialize_all_variables()
        return (loss, train, init)


    def pretrain(self, opt, X, iterations):
        loss, train, init = self.getOptimizationVariables(opt)
        
        self.sess.run(init)
        self.trained = True

        for it in range(iterations):

            if it % 100 == 0:
                print("Pretrain Iteration=", it)

            batch = self.samplePretrainBatch(X)
            self.sess.run(train, batch)


    def train(self, opt, X, iterations, subiterations):
        loss, train, init = self.getOptimizationVariables(opt)

        if not self.trained:
            self.sess.run(init)

        for it in range(iterations):

            if it % self.checkpoint_freq == 0:
                print("Checkpointing Train", it, self.checkpoint_file)
                self.save()

            batch = self.sampleBatch(X)
            
            for i in range(subiterations):
                self.sess.run(train, batch)




class TFNetworkModel(TFModel):
    """
    This class defines the abstract class for a tensorflow model for the primitives.
    """

    def __init__(self, 
                 statedim, 
                 actiondim, 
                 k):

        self.policy_networks = []
        self.transition_networks = []

        super(TFNetworkModel, self).__init__(statedim, actiondim, k)


    def getLossFunction(self):

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




        
