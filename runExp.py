#!/usr/bin/env python

from segmentcentroid.tfmodel.supervised_networks import *
import tensorflow as tf

from segmentcentroid.experiments import experiment1b as exp1
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

for i in range(0, 1):
    try:
        full_traj.append(j.plan())
        print(i)
    except:
        pass

opt = tf.train.AdamOptimizer(learning_rate=1e-2)

m = JHUJigSawsMultimodalModel(2)

m.train(opt, full_traj, 10, 1)

j.visualizePlans(full_traj, m, filename="resources/results/exp5-trajs7.png")
"""

"""
from segmentcentroid.planner.jigsaws_loader import JigsawsPlanner
from segmentcentroid.tfmodel.JHUJigSawsMultimodalModel import JHUJigSawsMultimodalModel
from segmentcentroid.tfmodel.JHUJigSawsModel import JHUJigSawsModel
import tensorflow as tf
from segmentcentroid.tfmodel.unsupervised_vision_networks import *
import matplotlib.pyplot as plt
import sys, traceback

j = JigsawsPlanner("/Users/sanjayk/Downloads/Knot_Tying/kinematics/AllGestures/", gtdirectory="/Users/sanjayk/Downloads/Knot_Tying/transcriptions/")
full_traj = []
ground_truth = []
sampling = 1

for i in range(0, 30):
    try:
        traj, gt = j.plan()
        full_traj.append([traj[i] for i in range(0,len(traj),sampling)])
        ground_truth.append([g/sampling for g in gt])
        print(i)
    except:
        print('-'*60)
        traceback.print_exc(file=sys.stdout)
        print('-'*60)


m = JHUJigSawsModel(6)

#opt = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
#m.pretrain(opt, full_traj, 100)

m.sess.run(tf.initialize_all_variables())

with tf.variable_scope("optimizer"):
    opt = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
    m.train(opt, full_traj, 1000, 1000)


j.visualizePlans([m.dataTransformer(t) for t in full_traj], m, filename="resources/results/exp5-trajs12.png", ground_truth=ground_truth)
"""

"""
ae = conv2LinearAutoencoder([120, 160, 3])

feed_dict = {}
feed_dict[ae['state']] = X[:100, :, :, :]

opt = tf.train.MomentumOptimizer(learning_rate=1e-2, momentum=0.01)
train = opt.minimize(ae['loss'])

m.sess.run(tf.initialize_all_variables())

for i in range(0,100):
    m.sess.run(train, feed_dict)
    print("Iteration",i,m.sess.run(ae['loss'], feed_dict))
"""


"""




m.train(opt, full_traj, 10, 1)

j.visualizePlans(full_traj, m, filename="resources/results/exp5-trajs7.png")
"""