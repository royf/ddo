#!/usr/bin/env python
from segmentcentroid.a3c.driver import train, collect_demonstrations
from segmentcentroid.tfmodel.AtariVisionModel import AtariVisionModel
from segmentcentroid.a3c.augmentedEnv import AugmentedEnv
from segmentcentroid.inference.forwardbackward import ForwardBackward
import tensorflow as tf

import ray

ray.init(num_cpus=2)

env, policy = train(2)
trajs = collect_demonstrations(env, policy)
a = AtariVisionModel(2)

with tf.variable_scope("optimizer2"):
    opt = tf.train.GradientDescentOptimizer(learning_rate=1e-3)
    a.sess.run(tf.initialize_all_variables())
    a.train(opt, trajs, 2, 2)

variables = ray.experimental.TensorFlowVariables(a.loss, a.sess)

weights = variables.get_weights()


a2 = AtariVisionModel(2)
a2.sess.run(tf.initialize_all_variables())
#env, policy = train(2, model=weights, k=2)
variables = ray.experimental.TensorFlowVariables(a2.loss, a2.sess)
variables.set_weights(weights)

#print(env.step(1))

