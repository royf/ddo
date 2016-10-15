#!/usr/bin/env python

from segmentcentroid.envs.GridWorldEnv import GridWorldEnv
from segmentcentroid.planner.mcts import MCTS
import numpy as np


##Example 1##
#Initialize GridWorld and query a state
MAP_NAME = 'resources/GridWorldMaps/4x5.txt'
gmap = np.loadtxt(MAP_NAME, dtype=np.uint8)
g = GridWorldEnv(gmap)
g.init()
print g.getState()
g.play(3)
print g.getState()


##Example 2##
g = GridWorldEnv(gmap)
m = MCTS(g)
print m.plan(15)
print m.plan(10, start=np.array([0, 3]))
