#!/usr/bin/env python
from segmentcentroid.a3c.driver import train, collect_demonstrations
from segmentcentroid.tfmodel.AtariVisionModel import AtariVisionModel
from segmentcentroid.a3c.augmentedEnv import AugmentedEnv
from segmentcentroid.inference.forwardbackward import ForwardBackward
import tensorflow as tf

import ray

ray.init()

with tf.Graph().as_default():
    a = AtariVisionModel(2)

    variables = ray.experimental.TensorFlowVariables(a.loss, a.sess)

    #env, policy = train(2)
    #trajs = collect_demonstrations(env, policy)

    with tf.variable_scope("optimizer2"):
        opt = tf.train.AdamOptimizer(learning_rate=1e-3)
        a.sess.run(tf.initialize_all_variables())
        #a.train(opt, trajs, 1000, 100)

    weights = variables.get_weights()

env, policy = train(12, model=weights, k=2, max_steps=10000)
trajs = collect_demonstrations(env, policy)


with tf.Graph().as_default():
    a = AtariVisionModel(2)

    variables = ray.experimental.TensorFlowVariables(a.loss, a.sess)

    with tf.variable_scope("optimizer2"):
        opt = tf.train.AdamOptimizer(learning_rate=1e-3)
        a.sess.run(tf.initialize_all_variables())
        a.train(opt, trajs, 1000, 10)

    weights = variables.get_weights()


env, policy = train(12, model=weights, k=2, policy=policy, max_steps=50000)

#print(env.step(1))

