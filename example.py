#!/usr/bin/env python

from segmentcentroid.envs.GridWorldEnv import GridWorldEnv
from segmentcentroid.planner.mcts import MCTS
from segmentcentroid.models.LogitModel import LogitModel

from segmentcentroid.inference.seginference import SegCentroidInferenceDiscrete

import numpy as np



##Example 1##
#Initialize GridWorld and query a state
"""
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
a = m.plan(8)
b = m.plan(5, start=np.array([2, 2]))
"""

#print a,b

a = [(np.array([3,0]), 1), (np.array([3,1]), 1), (np.array([3,2]), 1), (np.array([3,3]), 0), (np.array([0,3]), 3), (np.array([1,3]), 3), (np.array([2,3]), 3), (np.array([3,3]), 3)]
b = [(np.array([0,3]), 0), (np.array([1,3]), 0), (np.array([2,3]), 0), (np.array([3,3]), 0), (np.array([3,0]), 0), (np.array([3,1]), 0), (np.array([3,2]), 0), (np.array([3,3]), 0)]

#l = LogitModel(2,4)

#print l.theta
#print l.eval(np.matrix([1,1]).T)



s = SegCentroidInferenceDiscrete(LogitModel, 2)
print s.fit([a, b],2,4)

