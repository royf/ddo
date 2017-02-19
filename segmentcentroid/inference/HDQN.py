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

        self.augmentedsize = actiondim + model.k

        self.k = model.k

        self.model = model

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

        self.evalNetwork = self.createQNetwork()

        self.epsilon0 = epsilon0

        self.epsilon_decay_rate = epsilon_decay_rate

        self.resultsFile = None

        self.primtivesOnly = False

        #some nuisance logging parameters
        self.checkpoint_frequency = 100

        self.eval_frequency = 10

        self.eval_trials = 30

        self.update_frequency = 100


    def setResultsFile(self, resultsFile, resultInfo):
      self.resultsFile = resultsFile
      self.resultInfo = resultInfo

    def setPrimitivesOnly(self):
        self.primtivesOnly = True

    def createQNetwork(self):
        raise NotImplemented("Must provide a Q network")


    def observe(self, s):
        raise NotImplemented("Must provide an observation function")


    def translate(self, o):
      raise NotImplemented("Must provide a translate function")


    def eval(self, S, stable_weights=False):
        feedDict = {self.network['state']: S}
        out = self.sess.run(self.network['alloutput'], feedDict)

        if stable_weights:
          feedDict = {self.network['state']: S}
          out = self.sess.run(self.network['alloutput'], feedDict)
        else:
          feedDict = {self.evalNetwork['state']: S}
          out = self.sess.run(self.evalNetwork['alloutput'], feedDict)

        if self.primtivesOnly:
          out[:,0:self.actiondim] = -np.inf
          return out
        else:
          return out

    def argmax(self, S):
        return np.argmax(self.eval(S))

    def max(self, S, stable_weights):
        return np.max(self.eval(S, stable_weights), axis=1)


    def policy(self, saction, epsilon):
        if np.random.rand(1) < epsilon:

          if self.primtivesOnly:
            return np.random.choice(np.arange(self.actiondim, self.actiondim+self.k))
          else:
            return np.random.choice(np.arange(self.actiondim+self.k))

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
        A = np.zeros((size, self.augmentedsize))
        D = np.zeros((size, 1))
        R = np.zeros((size, 1))

        for i in range(size):
            S[i,:] = sample[i]['old_state']
            Sp[i,:] = sample[i]['new_state']
            A[i, sample[i]['action']] = 1 
            D[i,:] = sample[i]['done']
            R[i,:] = sample[i]['reward']

        V = np.reshape(self.max(Sp, True),(-1,1))
        Y = np.zeros((size, 1))

        for j in range(size):
            if D[j,0] == 1:
                Y[j, :] = R[j,:]
            else:
                Y[j, :] = R[j,:] + self.gamma*V[j,:]

        return S, A, Y


    def tieWeights(self):
      for i, v in enumerate(self.evalNetwork['weights']):
        self.sess.run(tf.assign(v, self.network['weights'][i]))


    def apply_primitive(self, i, remaining):
        prev_obs = self.observe(self.env.state)

        done = False

        primitive_reward = 0

        intermediate_states = [self.env.state]

        while not done:
            actions = np.eye(self.actiondim)

            remaining = remaining - 1

            
            #l = [ np.ravel(self.model.evalpi(i, [(self.env.state, actions[j,:])] ))  for j in self.env.possibleActions(self.env.state)]

            #chosen_action = self.env.possibleActions(self.env.state)[np.argmax(l)]

            l = np.array([ np.ravel(self.model.evalpi(i, [(self.env.state, actions[j,:])] ))  for j in range(self.actiondim)]).reshape(self.actiondim)
            l = np.abs(l)/np.sum(l)

            chosen_action = np.random.choice(np.arange(0, self.actiondim), size=1, p=l)

            
            reward = self.env.play(chosen_action)

            observation = self.observe(self.env.state)

            termination = (np.random.rand() < np.ravel(self.model.evalpsi(i, [(self.env.state, actions[1,:])])))

            done = self.env.termination or termination or remaining <= 0

            primitive_reward = primitive_reward + reward

            intermediate_states.append((self.env.state, chosen_action))

        return prev_obs, observation, primitive_reward, remaining, intermediate_states


    def apply_action(self, action):
        prev_obs = self.observe(self.env.state)
        reward = self.env.play(action)
        observation = self.observe(self.env.state)
        return prev_obs, observation, reward


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

            remaining_time = episodeLength

            while remaining_time > 0:

                action = self.argmax(observation)
                action = self.policy(action, epsilon)
                print("applied",action, remaining_time)

                if action >= self.actiondim:
                  #print(action)
                  prev_obs, observation, reward, remaining_time, _ = self.apply_primitive(action-self.actiondim, remaining_time)
                else:
                  prev_obs, observation, reward = self.apply_action(action)
                  remaining_time = remaining_time - 1

                if self.env.termination:
                    self.replay_buffer.append({'old_state': prev_obs, 
                                          'new_state': observation, 
                                          'action': action,
                                          'reward': reward,
                                           'done': True })

                else:
                    self.replay_buffer.append({'old_state': prev_obs, 
                                          'new_state': observation, 
                                          'action': action,
                                          'reward': reward,
                                          'done': False})


                S, A, Y = self.sample_batch(self.minibatch)

                self.sess.run(minimizer, {self.network['state']: S, 
                                       self.network['action']: A,
                                       self.network['y']: Y}) 

                if self.env.termination:
                  break

            self.results_array.append((self.env.reward, episodeLength-remaining_time, epsilon))

            print("Episode", episode, (reward, episodeLength-remaining_time, epsilon))

            if episode % self.update_frequency == (self.update_frequency - 1):
              self.tieWeights()

            if episode % self.eval_frequency == (self.eval_frequency-1):
              total_return = self.evalValueFunction(episodeLength, self.eval_trials)
              self.results_array.append((None, episode, total_return))
              print("Evaluating",episode, np.mean(total_return))


            if self.resultsFile != None \
                and episode % self.checkpoint_frequency == (self.checkpoint_frequency-1):
              print("Saving Data...")
              import pickle
              f = open(self.resultsFile, 'wb')
              pickle.dump({'data': self.results_array, 'info': self.resultInfo}, f)
              f.close()


    def evalValueFunction(self, episodeLength, trials):
        
        total_return = []

        for trial in range(trials):
          self.env.init()

          remaining_time = episodeLength
          observation = self.observe(self.env.state)

          while remaining_time > 0:
            action = self.argmax(observation)

            #print(action, remaining_time)

            if action >= self.actiondim:
                prev_obs, observation, reward, remaining_time, _ = self.apply_primitive(action-self.actiondim, remaining_time)
            else:
                prev_obs, observation, reward = self.apply_action(action)
                remaining_time = remaining_time - 1

            if self.env.termination:
              break

          total_return.append(self.env.reward)

        return total_return



class TabularHDQN(HDQN):

    def __init__(self,
                 env,
                 statedim,
                 actiondim,
                 model,
                 buffersize = 10000,
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

        a = tf.placeholder(tf.float32, shape=[None, self.augmentedsize])

        y = tf.placeholder(tf.float32, shape=[None, 1])

    
        table = tf.Variable(0*tf.random_uniform([1, self.statedim[0], self.statedim[1], self.augmentedsize]))

        inputx = tf.tile(tf.reshape(x, [-1, self.statedim[0], self.statedim[1], 1]), [1, 1, 1, self.augmentedsize])

        tiled_table = tf.tile(table, [tf.shape(x)[0],1,1,1])

        collapse = tf.reshape(tf.reduce_sum(tf.reduce_sum(tf.multiply(inputx, tiled_table), 1), 1), [-1, self.augmentedsize])

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

class LinearHDQN(HDQN):

    def __init__(self,
                 env,
                 statedim,
                 actiondim,
                 model,
                 regularization=0.01,
                 hidden_layer = 32,
                 buffersize = 100000,
                 gamma = 0.99,
                 learning_rate = 0.1,
                 minibatch=100,
                 epsilon0=1,
                 epsilon_decay_rate=1e-3):

        self.regularization = regularization
        self.hidden_layer = hidden_layer

        super(LinearHDQN, self).__init__(env, statedim, actiondim, model, buffersize, gamma, learning_rate, minibatch, epsilon0, epsilon_decay_rate)
        
        self.checkpoint_frequency = 1000
        self.eval_frequency = 20
        self.eval_trials = 10
        self.update_frequency = 2




    def createQNetwork(self):

        x = tf.placeholder(tf.float32, shape=[None, self.statedim[0]])

        a = tf.placeholder(tf.float32, shape=[None, self.augmentedsize])

        y = tf.placeholder(tf.float32, shape=[None, 1])

        W1 = tf.Variable(tf.random_normal([self.statedim[0], self.hidden_layer]))
        b1 = tf.Variable(tf.random_normal([ self.hidden_layer]))

        h = tf.nn.relu(tf.matmul(x, W1) + b1)

        W2 = tf.Variable(tf.random_normal([ self.hidden_layer, self.augmentedsize]))
        b2 = tf.Variable(tf.random_normal([self.augmentedsize]))

        alloutput = tf.reshape(tf.matmul(h, W2) + b2, [-1, self.augmentedsize])

        output = tf.reshape(tf.reduce_mean(tf.multiply(a, alloutput), 1), [-1, 1])

        weights = [W1, b1, W2, b2]

        woutput = tf.reduce_mean(tf.square(y-output)) #+  0.1*(tf.reduce_sum(tf.square(W)) + tf.reduce_sum(tf.square(b)))

        #for w in weights:
        woutput = woutput + self.regularization*(tf.reduce_sum(tf.square(alloutput)))
        
        return {'state': x, 
                'action': a, 
                'y': y,
                'output': output,
                'debug': h,
                'weights': weights,
                'alloutput': alloutput,
                'woutput': woutput}


    def observe(self, s):
        sp = np.reshape(s, (1,self.statedim[0]))
        return sp








