#!/usr/bin/env python

from segmentcentroid.envs.GridWorldEnv import GridWorldEnv
from segmentcentroid.planner.mcts import MCTS
from segmentcentroid.models.TabularModel import TabularModel
from segmentcentroid.models.LimitedTabularModel import LimitedTabularModel

from segmentcentroid.models.LogitModel import LogitModel, BinaryLogitModel

from segmentcentroid.models.TFModel import TFModel

from segmentcentroid.models.ForestModel import ForestModel

from segmentcentroid.planner.value_iteration import ValueIterationPlanner
from segmentcentroid.inference.seginference2 import JointSegCentroidInferenceDiscrete
from segmentcentroid.planner.traj_utils import *

import numpy as np
import copy

"""
This example demonstrates a proof of concept for manual segmentation
"""

#first we load the gridworld map and initialize the environment
MAP_NAME = 'resources/GridWorldMaps/Hallway.txt'
gmap = np.loadtxt(MAP_NAME, dtype=np.uint8)
full_traj = []
for i in range(0,100):
    g = GridWorldEnv(copy.copy(gmap), noise=0.0)
    g.generateRandomStartGoal()
    v = ValueIterationPlanner(g)
    traj = v.plan(max_depth=100)
    
    new_traj = []
    for t in traj:
        ns = np.ndarray(shape=(6))
        ns[0:2] = t[0]
        ns[2:4] = t[0]#np.argwhere(g.map == g.GOAL)[0]
        ns[4:6] = t[0]#np.argwhere(g.map == g.START)[0]

        new_traj.append((ns,t[1]))

    full_traj.append(new_traj)


g = GridWorldEnv(copy.copy(gmap), noise=0.0)

seg = JointSegCentroidInferenceDiscrete(TFModel, TabularModel, 8, 6, 4)
#s.fit(full_traj)

policies, transitions = seg.fit(full_traj, learning_rate=0.01, max_iters=10, max_liters=1)

#seg.fitTraj(full_traj[0], learning_rate=0.3, max_iters=10)

g = GridWorldEnv(copy.copy(gmap), noise=0.0)
g.generateRandomStartGoal()

#g.visualizePlan(full_traj, blank=True)

#Visualization Code
for i,p in enumerate(seg.policies):
    states = g.getAllStates()
    policy_hash = {}
    trans_hash = {}

    for s in states:
        
        ns = np.ndarray(shape=(6))
        ns[0:2] = s
        ns[2:4] = s#np.argwhere(g.map == g.GOAL)[0]
        ns[4:6] = s#np.argwhere(g.map == g.START)[0]

        if seg.transitions[i].visited(ns):

            action = np.argmax(p.eval(np.array(ns)))

            if p.eval(np.array(ns))[action] > .4: 
                policy_hash[s] = action

            #print(transitions[i].eval(np.array(ns)))
            trans_hash[s] = seg.transitions[i].eval(np.array(ns))

    g.visualizePolicy(policy_hash, trans_hash, blank=True)



"""

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




