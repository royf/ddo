#!/usr/bin/env python

from segmentcentroid.envs.GridWorldEnv import GridWorldEnv
from segmentcentroid.planner.mcts import MCTS
from segmentcentroid.models.LogitModel import LogitModel
from segmentcentroid.planner.value_iteration import ValueIterationPlanner
from segmentcentroid.inference.seginference import SegCentroidInferenceDiscrete, JointSegCentroidInferenceDiscrete
from segmentcentroid.planner.traj_utils import *

import numpy as np
import copy

"""
This example demonstrates a proof of concept for manual segmentation
"""

#first we load the gridworld map and initialize the environment
MAP_NAME = 'resources/GridWorldMaps/11x11-Rooms.txt'
gmap = np.loadtxt(MAP_NAME, dtype=np.uint8)



#We can query the planner by running
full_traj = []
for i in range(0,50):
    g = GridWorldEnv(copy.copy(gmap), noise=0.1)
    g.generateRandomStartGoal()
    v = ValueIterationPlanner(g)
    traj = v.plan(max_depth=100)
    full_traj.append(traj)

#g.visualizePlan(full_traj, blank=True)

s = SegCentroidInferenceDiscrete(LogitModel, 6)
q,p, policies = s.fit(full_traj,2,4)
print(p)


#Visualization Code
for p in policies:
    states = g.getAllStates()
    policy_hash = {}
    for s in states:
        policy_hash[s] = np.argmax(p.eval(np.array(s)))

    g.visualizePolicy(policy_hash)



"""

#Next, let's collect a dataset
#We can manually segment the data into each room's data
segments = []

traj = v.plan(max_depth=100)

segments.append(traj)

traj = v.plan(max_depth=100)

segments.append(traj)

traj = v.plan(max_depth=100)

segments.append(traj)

traj = v.plan(max_depth=100)

segments.append(traj)
"""

#We can fit the model with a logistic regression policy class

"""
s = JointSegCentroidInferenceDiscrete(LogitModel, LogitModel, 2)
q,p, policies = s.fit(traj,2,4)


#Visualization Code
for p in policies:
    states = g.getAllStates()
    policy_hash = {}
    for s in states:
        policy_hash[s] = np.argmax(p.eval(np.array(s)))

    g.visualizePolicy(policy_hash)
"""




