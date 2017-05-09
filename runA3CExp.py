#!/usr/bin/env python
from segmentcentroid.a3c.driver import train, collect_demonstrations
from segmentcentroid.tfmodel.AtariVisionModel import AtariVisionModel
from segmentcentroid.a3c.augmentedEnv import AugmentedEnv
from segmentcentroid.inference.forwardbackward import ForwardBackward
import tensorflow as tf

import ray
import gym

ray.init()

def runDDO(env_name="PongDeterministic-v3",
           num_options=2, 
           ddo_learning_rate=1e-3,
           steps_per_discovery=30000,
           rounds=3,
           num_demonstrations_per=100,
           ddo_max_iters=100,
           ddo_vq_iters=100,
           num_workers=12):

    g = tf.Graph()

    #initialize graph
    with g.as_default():
        a = AtariVisionModel(num_options, actiondim=(gym.make(env_name).action_space.n,1))
        variables = ray.experimental.TensorFlowVariables(a.loss, a.sess)

        with tf.variable_scope("optimizer2"):
            opt = tf.train.AdamOptimizer(learning_rate=ddo_learning_rate)
            a.sess.run(tf.initialize_all_variables())

    #run once to initialize
    env, policy = train(num_workers, env_name=env_name, model=weights, k=num_options, max_steps=100)
    trajs = collect_demonstrations(env, policy, N=num_demonstrations_per)



    for i in range(rounds):

        with g.as_default():

            with tf.variable_scope("optimizer2"):

                vq = ddo_vq_iters
                if i > 0:
                    vq = 0

                a.train(opt, trajs, ddo_max_iters, vq)

            weights = variables.get_weights()


        env, policy = train(num_workers, env_name=env_name, model=weights, k=num_options, max_steps=steps_per_discovery)
        trajs = collect_demonstrations(env, policy, N=num_demonstrations_per)


runDDO()
