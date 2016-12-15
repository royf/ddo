#!/usr/bin/env python

#from segmentcentroid.experiments import experiment1 as exp1
#exp1.runPolicies()

#from segmentcentroid.experiments import experiment2 as exp2
#exp2.runPolicies()

#from segmentcentroid.experiments import experiment3 as exp3
#exp3.runPolicies()

#from segmentcentroid.experiments import experiment4 as exp4
#exp4.runPolicies()

#from segmentcentroid.experiments import experiment5 as exp5
#exp5.runPolicies()


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