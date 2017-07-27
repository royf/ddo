#!/usr/bin/env python
from segmentcentroid.a3c.driver import train, collect_demonstrations
from segmentcentroid.tfmodel.AtariVisionModel import AtariVisionModel
from segmentcentroid.a3c.augmentedEnv import AugmentedEnv
from segmentcentroid.inference.forwardbackward import ForwardBackward
import tensorflow as tf

import ray
import gym

ray.init()
#MontezumaRevenge-v0
def runDDOI(env_name="PongDeterministic-v3",
           num_options=2, 
           ddo_learning_rate=1e-3,
           inital_steps=10,
           steps_per_discovery=10000,
           rounds=1,
           num_demonstrations_per=100,
           ddo_max_iters=100,
           ddo_vq_iters=100,
           num_workers=1):

    g = tf.Graph()

    #initialize graph
    with g.as_default():
        a = AtariVisionModel(num_options, actiondim=(gym.make(env_name).action_space.n,1))
        variables = ray.experimental.TensorFlowVariables(a.loss, a.sess)

        with tf.variable_scope("optimizer2"):
            opt = tf.train.AdamOptimizer(learning_rate=ddo_learning_rate)
            a.sess.run(tf.initialize_all_variables())

        weights = variables.get_weights()

    #initialize with intrinsic motivation
    env, policy = train(num_workers, env_name=env_name, model=weights, k=num_options, max_steps=inital_steps, intrinsic=True)
    trajs,_ = collect_demonstrations(env, policy, N=num_demonstrations_per, epLengthProxy=True)

    for i in range(rounds):

        with g.as_default():

            with tf.variable_scope("optimizer2"):

                vq = ddo_vq_iters
                if i > 0:
                    vq = 0

                a.train(opt, trajs, ddo_max_iters, vq)

            weights = variables.get_weights()


        env, policy = train(num_workers, policy=policy, env_name=env_name, model=weights, k=num_options, max_steps=steps_per_discovery)
        trajs, reward = collect_demonstrations(env, policy, N=num_demonstrations_per)

    return {'reward': reward, 'env': env_name, 'num_options': num_options, 'ddo': True, 'intrinsic': True}


def runDDO(env_name="PongDeterministic-v3",
           num_options=2, 
           ddo_learning_rate=1e-3,
           steps_per_discovery=10000,
           rounds=1,
           num_demonstrations_per=100,
           ddo_max_iters=100,
           ddo_vq_iters=100,
           num_workers=1):

    g = tf.Graph()

    #initialize graph
    with g.as_default():
        a = AtariVisionModel(num_options, actiondim=(gym.make(env_name).action_space.n,1))
        variables = ray.experimental.TensorFlowVariables(a.loss, a.sess)

        with tf.variable_scope("optimizer2"):
            opt = tf.train.AdamOptimizer(learning_rate=ddo_learning_rate)
            a.sess.run(tf.initialize_all_variables())

        weights = variables.get_weights()

    #run once to initialize
    env, policy = train(num_workers, env_name=env_name, model=weights, k=num_options, max_steps=1, intrinsic=False)
    trajs,_ = collect_demonstrations(env, policy, N=num_demonstrations_per)

    for i in range(rounds):

        with g.as_default():

            with tf.variable_scope("optimizer2"):

                vq = ddo_vq_iters
                if i > 0:
                    vq = 0

                a.train(opt, trajs, ddo_max_iters, vq)

            weights = variables.get_weights()


        env, policy = train(num_workers, policy=policy, env_name=env_name, model=weights, k=num_options, max_steps=steps_per_discovery)
        trajs, reward = collect_demonstrations(env, policy, N=num_demonstrations_per)

    return {'reward': reward, 'env': env_name, 'num_options': num_options, 'ddo': True, 'intrinsic': False}


def runA3C(env_name="PongDeterministic-v3",
           steps=1000,
           num_demonstrations_per=100,
           num_workers=1):

    #run once to initialize
    env, policy = train(num_workers, env_name=env_name, max_steps=steps)
    trajs, reward = collect_demonstrations(env, policy, N=num_demonstrations_per)

    return {'reward': reward, 'env': env_name, 'ddo': False}



def runAll(num_steps, envs, num_workers, outputfile='out.p'):
    import pickle
    f = open(outputfile, 'wb')

    output = []

    for e in envs:
        print("Running", e)

        if num_steps >= 5000:
            rounds = int(num_steps/9999)
        else:
            rounds = 1
        
        data = (runA3C(env_name=e, steps=num_steps, num_workers=num_workers),
                     runDDO(env_name=e, steps_per_discovery=num_steps, rounds=rounds, num_workers=num_workers),
                     runDDOI(env_name=e, steps_per_discovery=num_steps, rounds=rounds, num_workers=num_workers))

        print("###Data", data)
        output.append(data)

    pickle.dump(output, f)


runAll(600000, ['PongDeterministic-v3'], 1)

