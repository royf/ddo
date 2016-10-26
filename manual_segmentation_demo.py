#!/usr/bin/env python

from segmentcentroid.envs.GridWorldEnv import GridWorldEnv
from segmentcentroid.planner.mcts import MCTS
from segmentcentroid.models.LogitModel import LogitModel
from segmentcentroid.planner.value_iteration import ValueIterationPlanner
from segmentcentroid.inference.seginference import SegCentroidInferenceDiscrete
from segmentcentroid.planner.traj_utils import *

import numpy as np

"""
This example demonstrates a proof of concept for manual segmentation
"""

#first we load the gridworld map and initialize the environment
MAP_NAME = 'resources/GridWorldMaps/11x11-Rooms.txt'
gmap = np.loadtxt(MAP_NAME, dtype=np.uint8)
g = GridWorldEnv(gmap, noise=0.3)


#Then we initialize the planner, and visualize the policy 
v = ValueIterationPlanner(g)
g.visualizePolicy(v.policy)


#We can query the planner by running
traj = v.plan(max_depth=100)
g.visualizePlan(traj)


#Next, let's collect a dataset
#We can manually segment the data into each room's data
segments = []

traj = v.plan(max_depth=100)

segments.extend(waypoint_segment(traj, [(2,5),  (8,4), (5,7), (3,9)]))

traj = v.plan(max_depth=100)

segments.extend(waypoint_segment(traj, [(2,5),  (8,4), (5,7), (3,9)]))

traj = v.plan(max_depth=100)

segments.extend(waypoint_segment(traj, [(2,5),  (8,4), (5,7), (3,9)]))

traj = v.plan(max_depth=100)

segments.extend(waypoint_segment(traj, [(2,5),  (8,4), (5,7), (3,9)]))



#We can fit the model with a logistic regression policy class
s = SegCentroidInferenceDiscrete(LogitModel, 2)
q,p, policies = s.fit(segments,2,4)


#Visualization Code
for p in policies:
    states = g.getAllStates()
    policy_hash = {}
    for s in states:
        policy_hash[s] = np.argmax(p.eval(np.array(s)))

    g.visualizePolicy(policy_hash)


