#!/usr/bin/env python

from segmentcentroid.experiments import experiment1 as exp1
exp1.runPolicies()

#from segmentcentroid.experiments import experiment2 as exp2
#exp2.runPolicies()

#from segmentcentroid.experiments import experiment3 as exp3
#exp3.runPolicies()

#from segmentcentroid.experiments import experiment4 as exp4
#exp4.runPolicies()

"""
import numpy as np
import tensorflow as tf
from segmentcentroid.planner.jigsaws_loader import JigsawsPlanner

j = JigsawsPlanner('/Users/sanjayk/Downloads/Knot_Tying/kinematics/AllGestures/')
full_traj = []
for i in range(0,20):
    full_traj.append(j.plan())

from segmentcentroid.tfmodel.JHUJigSawsModel import JHUJigSawsModel
m  = JHUJigSawsModel(4)


opt = tf.train.AdamOptimizer(learning_rate=1e-2)
m.pretrain(opt, full_traj, 1000)

m.train(opt, full_traj, 10, 1)
"""