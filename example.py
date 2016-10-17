#!/usr/bin/env python

from segmentcentroid.envs.GridWorldEnv import GridWorldEnv
from segmentcentroid.planner.mcts import MCTS
from segmentcentroid.models.SoftMaxLogit import SoftMaxLogit

from segmentcentroid.inference.seginference import SegCentroidInferenceDiscrete

import numpy as np

import theano
import theano.tensor as T


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
a = m.plan(8)
b = m.plan(5, start=np.array([2, 2]))

print a,b



#s = SegCentroidInferenceDiscrete(SoftMaxLogit, 4)
#s.fit([a, b],2,4)

