#!/usr/bin/env python

from segmentcentroid.envs.GridWorldEnv import GridWorldEnv
from segmentcentroid.planner.mcts import MCTS
from segmentcentroid.models.LogitModel import LogitModel
from segmentcentroid.planner.value_iteration import ValueIterationPlanner

from segmentcentroid.inference.seginference import SegCentroidInferenceDiscrete

import numpy as np

MAP_NAME = 'resources/GridWorldMaps/11x11-Rooms.txt'
gmap = np.loadtxt(MAP_NAME, dtype=np.uint8)
g = GridWorldEnv(gmap, noise=0.3)

#g.visualize(policy)

v = ValueIterationPlanner(g)

g.visualizePolicy(v.policy)

traj = v.plan(max_depth=100)

g.visualizePlan(traj)

traj = v.plan(max_depth=100)

g.visualizePlan(traj)

traj = v.plan(max_depth=100)

g.visualizePlan(traj)







##Example 1##
#Initialize GridWorld and query a state
"""

g.init()
print g.getState()
g.play(3)
print g.getState()
"""


