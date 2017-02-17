"""
This module implements a deep q network in tensorflow
"""
import tensorflow as tf
import numpy as np
import copy

class DQN(object):

    def __init__(self,
                 env,
                 statedim,
                 actiondim,
                 buffersize = 100000,
                 gamma = 0.99,
                 learning_rate = 0.1,
                 minibatch=100,
                 epsilon0=1,
                 epsilon_decay_rate=1e-3):

        self.statedim = statedim
        self.actiondim = actiondim

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

        #some nuisance logging parameters
        self.checkpoint_frequency = 100

        self.eval_frequency = 10

        self.eval_trials = 30

        self.update_frequency = 100



    def setResultsFile(self, resultsFile, resultInfo):
      self.resultsFile = resultsFile
      self.resultInfo = resultInfo

    def createQNetwork(self):
        raise NotImplemented("Must provide a Q network")


    def observe(self, s):
        raise NotImplemented("Must provide an observation function")

    def eval(self, S, stable_weights=False):
        
        if stable_weights:
          feedDict = {self.network['state']: S}
          out = self.sess.run(self.network['alloutput'], feedDict)
          return out
        else:
          feedDict = {self.evalNetwork['state']: S}
          out = self.sess.run(self.evalNetwork['alloutput'], feedDict)
          return out

    def argmax(self, S):
        #print("amax", np.argmax(self.eval(S)), self.eval(S))
        return np.argmax(self.eval(S))

    def max(self, S, stable_weights):
        #print(S, np.max(self.eval(S), axis=1))
        return np.max(self.eval(S, stable_weights), axis=1)


    def policy(self, saction, epsilon):
        if np.random.rand(1) < epsilon:
            return np.random.choice(np.arange(self.actiondim))
        else:
            return saction

    def tieWeights(self):
      for i, v in enumerate(self.evalNetwork['weights']):
        self.sess.run(tf.assign(v, self.network['weights'][i]))

    def sample_batch(self, size):
        N = len(self.replay_buffer)
        indices = np.random.choice(np.arange(N), size)
        sample = [self.replay_buffer[i] for i in indices]

        sarraydims = copy.copy(self.sarraydims)
        sarraydims.insert(0, size)

        S = np.zeros(sarraydims)
        Sp = np.zeros(sarraydims)
        A = np.zeros((size, self.actiondim))
        D = np.zeros((size, 1))
        R = np.zeros((size, 1))

        for i in range(size):
            S[i,:] = sample[i]['old_state']
            Sp[i,:] = sample[i]['new_state']
            A[i,sample[i]['action']] = 1 
            D[i,:] = sample[i]['done']
            R[i,:] = sample[i]['reward']

        V = np.reshape(self.max(Sp, True),(-1,1))
        V1 = np.reshape(self.max(Sp, False),(-1,1))

        #print(A)

        #print(V,V1)
        #print(V)
        Y = np.zeros((size, 1))

        for j in range(size):
            if D[j,0] == 1:
                Y[j, :] = R[j,:]
            else:
                Y[j, :] = R[j,:] + self.gamma*V[j,:]


        #print(Y)

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

                #print(self.env.state)

                action = self.argmax(observation)

                action = self.policy(action, epsilon)

                #print(action)

                prev_obs = observation

                reward = self.env.play(action)

                observation = self.observe(self.env.state)

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

            #S, A, Y = self.sample_batch(self.minibatch)

                S, A, Y = self.sample_batch(self.minibatch)
            #for i in range(1):

                #batch_loss = self.sess.run(self.network['woutput'], {self.network['state']: S, self.network['action']: A, self.network['y']: Y})
                #print("before", batch_loss, batch_loss.shape)

            #for i in range(1,100):
            #S, A, Y = self.sample_batch(self.minibatch)
              #print(self.network['y'])
                self.sess.run(minimizer, {self.network['state']: S, 
                                      self.network['action']: A,
                                       self.network['y']: Y}) 

                batch_loss = self.sess.run(self.network['debug'], {self.network['state']: S, self.network['action']: A, self.network['y']: Y}) #list(zip(S, self.sess.run(self.network['output'], {self.network['state']: S, self.network['action']: A, self.network['y']: Y}), self.sess.run(self.network['y'], {self.network['state']: S, self.network['action']: A, self.network['y']: Y})))
                #print("after",batch_loss, A)

                if self.env.termination:
                  break




            print("Episode", episode, (self.env.reward, step, epsilon))
            self.results_array.append((self.env.reward, step, epsilon))


            if episode % self.update_frequency == (self.update_frequency - 1):
              self.tieWeights()
            #  target_weights = zip(self.network['weights'], self.sess.run(self.network['weights']))


            if episode % self.eval_frequency == (self.eval_frequency-1):
              total_return = self.evalValueFunction(episodeLength, self.eval_trials)
              self.results_array.append((None, episode, total_return))
              print("Evaluating",episode, np.mean(total_return), total_return)


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

            print(self.eval(observation), action)

            self.env.play(action)

            observation = self.observe(self.env.state)
            
            remaining_time = remaining_time - 1

            if self.env.termination:
              break

          total_return.append(self.env.reward)

        return total_return




    def sampleTrajectories(self, episodeLength, trials):
        
        trajectories = []

        for trial in range(trials):
          self.env.init()

          remaining_time = episodeLength
          observation = self.observe(self.env.state)

          traj = []

          while remaining_time > 0:
            action = self.argmax(observation)
            av = np.zeros((self.actiondim, 1)) 
            av[action] = 1

            #print(self.env.state, action, self.eval(observation))

            traj.append((observation[0,:] , av))

            self.env.play(action)

            observation = self.observe(self.env.state)
            
            remaining_time = remaining_time - 1

            if self.env.termination:
              break

          trajectories.append(traj)

        return trajectories


class TabularDQN(DQN):

    def __init__(self,
                 env,
                 statedim,
                 actiondim,
                 buffersize = 100000,
                 gamma = 0.99,
                 learning_rate = 0.1,
                 minibatch=100,
                 epsilon0=1,
                 epsilon_decay_rate=1e-3):

        super(TabularDQN, self).__init__(env, statedim, actiondim, buffersize, gamma, learning_rate, minibatch, epsilon0, epsilon_decay_rate)


    def createQNetwork(self):
        sarraydims = copy.copy(self.sarraydims)
        sarraydims.insert(0, None)

        x = tf.placeholder(tf.float32, shape=sarraydims)

        a = tf.placeholder(tf.float32, shape=[None, self.actiondim])

        y = tf.placeholder(tf.float32, shape=[None, 1])

    
        table = tf.Variable(0.0*tf.random_uniform([1, self.statedim[0], self.statedim[1], self.actiondim]))

        inputx = tf.tile(tf.reshape(x, [-1, self.statedim[0], self.statedim[1], 1]), [1, 1, 1, self.actiondim])

        tiled_table = tf.tile(table, [tf.shape(x)[0],1,1,1])

        collapse = tf.reshape(tf.reduce_sum(tf.reduce_sum(tf.multiply(inputx, tiled_table), 1), 1), [-1, self.actiondim])

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