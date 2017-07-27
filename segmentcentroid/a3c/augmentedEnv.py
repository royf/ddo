import gym
from gym import error, spaces, utils
from gym.utils import seeding
import cv2
import numpy as np
import ray
from segmentcentroid.tfmodel.AtariVisionModel import AtariVisionModel
import tensorflow as tf

class AugmentedEnv(gym.Env):
  metadata = {'render.modes': ['human']}
  

  def __init__(self, gymEnvName, model_weights, k, intrinsic=False):

    self.env = gym.make(gymEnvName)

    g = tf.Graph()
    with g.as_default():
        model = AtariVisionModel(k,actiondim=(self.env.action_space.n,1))
        model.sess.run(tf.initialize_all_variables())
        variables = ray.experimental.TensorFlowVariables(model.loss, model.sess)

        variables.set_weights(model_weights)

    self.model = model
    self.trajectory_set = set()
    self.intrinsic = intrinsic

    self.real_action_space = self.env.action_space.n
    
    #print(self.real_action_space)

    self.action_space = spaces.Discrete(self.env.action_space.n + model.k) 
    #print("####",self.env.action_space.n + model.k)
    self.obs = None
    self.done = False
    self.spec = self.env.spec

  def _process_frame42(self, frame):
    frame = frame[34:(34+160), :160]
    # Resize by half, then down to 42x42 (essentially mipmapping). If
    # we resize directly we lose pixels that, when mapped to 42x42,
    # aren't close enough to the pixel boundary.
    frame = cv2.resize(frame, (80, 80))
    frame = cv2.resize(frame, (42, 42))
    frame = frame.mean(2)
    frame = frame.astype(np.float32)
    frame *= (1.0 / 255.0)
    frame = np.reshape(frame, [42, 42, 1])
    return frame


  def _process_frameTuple(self, frame, discretization=32.0, rep=16):
    frame = frame[34:(34+160), :160]
    # Resize by half, then down to 42x42 (essentially mipmapping). If
    # we resize directly we lose pixels that, when mapped to 42x42,
    # aren't close enough to the pixel boundary.
    frame = cv2.resize(frame, (80, 80))
    frame = cv2.resize(frame, (40, 40))
    frame = cv2.resize(frame, (20, 20))
    frame = cv2.resize(frame, (rep, rep))
    frame = frame.mean(2)
    frame = frame.astype(np.float32)
    frame = np.round(frame / discretization)
    frame = np.reshape(frame, (rep*rep,)).tolist()
    return tuple(frame)

  def _step(self, action):
    N = self.env.action_space.n
    actions = np.eye(N)
    
    if action < N:
    
        self.obs, reward, self.done, info = self.env._step(action)

        if self.intrinsic:

            tup = self._process_frameTuple(self.obs)
            
            if tup in self.trajectory_set:
                reward = 0
            else:
                reward = 1

            self.trajectory_set.add(tup)

    else:
        #obs = cv2.Canny(self.obs,100,200)
        #from matplotlib import pyplot as plt
        #plt.imshow(self.obs,cmap = 'gray')
        #plt.show()
        proc_obs = self._process_frame42(self.obs)

        done = self.done
        term = False
        reward = 0

        while (not done):  
            l = [ self.model.evalpi(action-N, [(proc_obs, actions[j,:])])[0]  for j in range(N)]
            #self.obs, rewardl, self.done, info = self._step(np.random.choice(np.arange(0,N),p=l/np.sum(l)))
            self.obs, rewardl, self.done, info = self._step(np.argmax(l))

            reward = reward + rewardl

            obs = self.obs
            done = self.done

            if (np.random.rand(1) > np.maximum(np.ravel(self.model.evalpsi(int(action-N), [(proc_obs, actions[1,:])])), 0.1)):
                #print("break")
                break

            #print(term)

    return self.obs, reward, self.done, info

  def _reset(self):
    self.obs = self.env._reset()
    self.done = False
    return self.obs

  def _render(self, mode='human', close=False):
    return self.env._render(mode, close)
