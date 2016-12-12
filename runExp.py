#!/usr/bin/env python

from segmentcentroid.experiments import experiment1 as exp1
exp1.runPolicies()

#from segmentcentroid.experiments import experiment2 as exp2
#exp2.runPolicies()

#from segmentcentroid.experiments import experiment3 as exp3
#exp3.runPolicies()

#from segmentcentroid.experiments import experiment4 as exp4
#exp4.runPolicies()

#from segmentcentroid.experiments import experiment5 as exp5
#exp5.runPolicies()

"""
from segmentcentroid.planner.jigsaws_loader import JigsawsPlanner
from segmentcentroid.tfmodel.JHUJigSawsMultimodalModel import JHUJigSawsMultimodalModel
import tensorflow as tf

j = JigsawsPlanner("/Users/sanjayk/Downloads/Knot_Tying/kinematics/AllGestures/", vdirectory="/Users/sanjayk/Downloads/Knot_Tying/video/")
full_traj = []

full_traj.append(j.plan())

opt = tf.train.AdamOptimizer(learning_rate=1e-2)

j = JHUJigSawsMultimodalModel(1)

j.pretrain(opt, full_traj, 100)
"""