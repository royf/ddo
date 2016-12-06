#!/usr/bin/env python

from segmentcentroid.experiments import experiment1 as exp1
exp1.runPolicies()

#from segmentcentroid.experiments import experiment2 as exp2
#exp2.runPolicies()

#from segmentcentroid.experiments import experiment3 as exp3
#exp3.runPolicies()

#from segmentcentroid.experiments import experiment4 as exp4
#exp4.runPolicies()

import numpy as np
import tensorflow as tf
from segmentcentroid.planner.jigsaws_loader import JigsawsPlanner

j = JigsawsPlanner('/Users/sanjayk/Downloads/Knot_Tying/kinematics/AllGestures/')
full_traj = []
for i in range(0,20):
    full_traj.append(j.plan())

sdim = 37

adim = 37

from segmentcentroid.tfmodel.JHUJigSawsModel import JHUJigSawsModel
m  = JHUJigSawsModel((sdim,1), (adim,1), 4)


opt = tf.train.AdamOptimizer(learning_rate=1e-2)
loss = m.getLossFunction()[0]
train = opt.minimize(loss)
init = tf.initialize_all_variables()


m.sess.run(init)
for i in range(100):
    batch = m.samplePretrainBatch(full_traj)
    m.sess.run(train, batch)


for it in range(5):
    print("Iteration",it)
    batch = m.sampleBatch(full_traj)
    for i in range(1):
        m.sess.run(train, batch)

for t in full_traj:
    m.fb.fit([t])
    j.visualizePlan(t, segmentation=np.argmax(m.fb.Q, axis=1))















