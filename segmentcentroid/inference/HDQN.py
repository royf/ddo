"""
This module implements a hierarchical deep q network in tensorflow
basically is a DQN with k*a actions
"""
import tensorflow as tf
import numpy as np
import copy

class HDQN(object):

    def __init__(self,
                 env,
                 statedim,
                 actiondim,
                 model,
                 buffersize = 100000,
                 gamma = 0.99,
                 learning_rate = 0.1,
                 minibatch=100,
                 epsilon0=1,
                 epsilon_decay_rate=1e-3):

        self.statedim = statedim
        self.actiondim = actiondim

        self.k = model.k

        self.env = env

        self.sarraydims = [s for s in statedim]

        self.buffersize = buffersize
        self.replay_buffer = []

        self.gamma = gamma
        self.learning_rate = learning_rate

        self.minibatch = minibatch

        self.results_array = []

        self.sess = tf.Session()

        self.network = self.createQNetwork()

        self.epsilon0 = epsilon0

        self.epsilon_decay_rate = epsilon_decay_rate



    def createQNetwork(self):
        raise NotImplemented("Must provide a Q network")


    def observe(self, s):
        raise NotImplemented("Must provide an observation function")


    def translate(self, o):
      raise NotImplemented("Must provide a translate function")




    #unpacks into action and hint
    def unpack(self, res):
        return res - int(res/self.actiondim), int(res/self.actiondim)

    def pack(self, a, h):
        #print(int((float(h)/self.k)*self.actiondim) + a)
        return int((float(h)/self.k)*self.actiondim) + a


    def eval(self, S):
        feedDict = {self.network['state']: S}
        return self.sess.run(self.network['alloutput'], feedDict)

    def argmax(self, S):
        return np.argmax(self.eval(S), axis=1)

    def max(self, S):
        return np.max(self.eval(S), axis=1)


    def policy(self, saction, epsilon):
        if np.random.rand(1) < epsilon:
            return np.random.choice(np.arange(self.actiondim))
        else:
            return saction


    def sample_batch(self, size):
        N = len(self.replay_buffer)
        indices = np.random.choice(np.arange(N), size)
        sample = [self.replay_buffer[i] for i in indices]

        sarraydims = copy.copy(self.sarraydims)
        sarraydims.insert(0, size)

        S = np.zeros(sarraydims)
        Sp = np.zeros(sarraydims)
        A = np.zeros((size, self.actiondim*self.k))
        D = np.zeros((size, 1))
        R = np.zeros((size, 1))

        for i in range(size):
            S[i,:] = sample[i]['old_state']
            Sp[i,:] = sample[i]['new_state']
            A[i, self.pack(sample[i]['action'], sample[i]['hint']) ] = 1 
            D[i,:] = sample[i]['done']
            R[i,:] = sample[i]['reward']

        V = np.reshape(self.max(Sp),(-1,1))
        Y = np.zeros((size, 1))

        for j in range(size):
            if D[j,0] == 1:
                Y[j, :] = R[j,:]
            else:
                Y[j, :] = R[j,:] + self.gamma*V[j,:]

        return S, A, Y


    def train(self, episodes, episodeLength):
        opt = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
        self.sess.run(tf.initialize_all_variables())
        minimizer = opt.minimize(self.network['woutput'])

        for episode in range(episodes):

            if len(self.replay_buffer) > self.buffersize:
                self.replay_buffer = self.replay_buffer[-self.buffersize:]


            self.env.init()

            observation = self.observe(self.env.state)

            epsilon = self.epsilon0/(episode*self.epsilon_decay_rate+1)

            for step in range(episodeLength):

                action, hint = self.unpack(self.argmax(observation)[0])
                action = self.policy(action, epsilon)

                prev_obs = observation

                reward = self.env.play(action)

                observation = self.observe(self.env.state)

                if self.env.termination:
                    self.replay_buffer.append({'old_state': prev_obs, 
                                          'new_state': observation, 
                                          'action': action,
                                          'hint': hint,
                                          'reward': reward,
                                           'done': True })
                    break

                else:
                    self.replay_buffer.append({'old_state': prev_obs, 
                                          'new_state': observation, 
                                          'action': action,
                                          'hint': hint,
                                          'reward': reward,
                                          'done': False})

            print("Episode", episode, (reward, step, epsilon))
            self.results_array.append((reward, step, epsilon))

            S, A, Y = self.sample_batch(self.minibatch)

            self.sess.run(minimizer, {self.network['state']: S, 
                                   self.network['action']: A,
                                   self.network['y']: Y}) 



class TabularHDQN(HDQN):

    def __init__(self,
                 env,
                 statedim,
                 actiondim,
                 model,
                 buffersize = 1e5,
                 gamma = 0.99,
                 learning_rate = 0.1,
                 minibatch=100,
                 epsilon0=1,
                 epsilon_decay_rate=1e-3):

        super(TabularHDQN, self).__init__(env, statedim, actiondim, model, buffersize, gamma, learning_rate, minibatch, epsilon0, epsilon_decay_rate)


    def createQNetwork(self):
        sarraydims = copy.copy(self.sarraydims)
        sarraydims.insert(0, None)

        x = tf.placeholder(tf.float32, shape=sarraydims)

        a = tf.placeholder(tf.float32, shape=[None, self.actiondim*self.k])

        y = tf.placeholder(tf.float32, shape=[None, 1])

    
        table = tf.Variable(0*tf.random_uniform([1, self.statedim[0], self.statedim[1], self.actiondim*self.k]))

        inputx = tf.tile(tf.reshape(x, [-1, self.statedim[0], self.statedim[1], 1]), [1, 1, 1, self.actiondim*self.k])

        tiled_table = tf.tile(table, [tf.shape(x)[0],1,1,1])

        collapse = tf.reshape(tf.reduce_sum(tf.reduce_sum(tf.multiply(inputx, tiled_table), 1), 1), [-1, self.actiondim*self.k])

        output = tf.reshape(tf.reduce_mean(tf.multiply(a, collapse), 1), [-1, 1])

        woutput = tf.reduce_mean((y-output)**2)
        
        return {'state': x, 
                'action': a, 
                'y': y,
                'output': output,
                'alloutput': collapse,
                'woutput': woutput}


    def observe(self, s):
        sarraydims = copy.copy(self.sarraydims)
        sarraydims.insert(0, 1)

        sp = np.zeros(shape=sarraydims)
        sp[0, s[0],s[1]] =  1
        return sp










