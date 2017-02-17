"""
This class defines a wrapper for all
the open ai gym atari environments
"""

import gym
import numpy as np
from .AbstractEnv import *

class AtariEnv(AbstractEnv):

    def __init__(self, env_name):

        self.env = gym.make(env_name)

        self.actions = self.env.action_space.n

        super(AtariEnv, self).__init__()
    """
    This function initializes the envioronment
    """
    def init(self, state=None, time=0, reward=0):
        self.state = self.env.reset()
        self.reward = 0
        self.time = 0
        self.termination = False

    """
    This function returns the current state, time, total reward, and termination
    """
    def getState(self):
        return self.state, self.time, self.reward, self.termination


    """
    This function takes an action
    """
    def play(self, action):
        self.state, reward, self.termination, _ = self.env.step(action)
        self.reward = self.reward + reward
        return reward

    """
    This function determins the possible actions at a state s, if none provided use 
    current state
    """
    def possibleActions(self, s=None):
        return np.arange(0, self.actions)


    """
    This function rolls out a policy which is a map from state to action
    """
    def rollout(self, policy):
        trajectory = []

        while not self.terminated:
            self.play(policy(self.state))
            trajectory.append(self.getState())

        return trajectory


        

     
