#!/usr/bin/env python
from segmentcentroid.a3c.driver import train, collect_demonstrations
from segmentcentroid.tfmodel.AtariVisionModel import AtariVisionModel
import tensorflow as tf

env, policy = train(2)
trajs = collect_demonstrations(env, policy)
a = AtariVisionModel(2)

with tf.variable_scope("optimizer"):
    opt = tf.train.GradientDescentOptimizer(learning_rate=1e-3)
    a.sess.run(tf.initialize_all_variables())
    a.train(opt, trajs, 10, 10)

