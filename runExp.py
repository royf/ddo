#!/usr/bin/env python
from segmentcentroid.envs.GridWorldWallsEnv import GridWorldWallsEnv
import numpy as np
import copy
from segmentcentroid.planner.value_iteration import ValueIterationPlanner
from segmentcentroid.planner.traj_utils import *
import distance

import tensorflow as tf
import tensorflow.contrib.slim as slim

MAP_NAME = 'resources/GridWorldMaps/experiment1.txt'
gmap = np.loadtxt(MAP_NAME, dtype=np.uint8)
N = 20

"""
generate string
"""
def traj_2_string(traj):
    s = ''
    for t in traj:
        s += str(t[1])
    return s


"""
measure distance between trajectories
"""
def traj_metric(t1, t2):
    return distance.levenshtein(t1, t2)

def encode(m):
    shape = (m.shape[0], m.shape[1], 4)
    M = np.zeros(shape)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            M[i,j,m[i,j]] = 1

    return M



trajs = []

for i in range(N):
    g = GridWorldWallsEnv(copy.copy(gmap), noise=0.0)
    g.generateRandomStartGoal()
    v = ValueIterationPlanner(g)
    traj = v.plan(max_depth=100)

    trajs += [(encode(g.map), traj_2_string(traj))]

X1 = np.zeros((N*N, 8, 9, 4))
X2 = np.zeros((N*N, 8, 9, 4))
Y = np.zeros((N*N, 1))

c = 0
for i in range(N):
    for j in range(N):
        dist = np.exp(-traj_metric(trajs[i][1], trajs[j][1] )/5)
        X1[c,:,:,:] = trajs[i][0]
        X2[c,:,:,:] = trajs[i][1]
        Y[c] = dist
        c += 1  

#print(trajs)

context_shape = [None] + list(trajs[0][0].shape)

C1 = tf.placeholder(tf.float32, shape=context_shape)
C2 = tf.placeholder(tf.float32, shape=context_shape)
D = tf.placeholder(tf.float32, shape=[None,1])


#first layer
conv_k1 = tf.Variable(tf.truncated_normal([3, 3, 4, 64], dtype=tf.float32), name='weights')
conv_b1 = tf.Variable(tf.truncated_normal(shape=[64], dtype=tf.float32), trainable=True, name='biases')

#c1
c1_conv_ker = tf.nn.conv2d(C1, conv_k1, [1, 3, 3, 1], padding='SAME')
c1_bias = tf.nn.bias_add(c1_conv_ker, conv_b1)
c1_conv1 = tf.nn.sigmoid(c1_bias, name='c1c1')


#c2
c2_conv_ker = tf.nn.conv2d(C2, conv_k1, [1, 3, 3, 1], padding='SAME')
c2_bias = tf.nn.bias_add(c2_conv_ker, conv_b1)
c2_conv1 = tf.nn.sigmoid(c2_bias, name='c1c2')




#two layer
conv_k2 = tf.Variable(tf.truncated_normal([3, 3, 64, 32], dtype=tf.float32), name='weights')
conv_b2 = tf.Variable(tf.truncated_normal(shape=[32], dtype=tf.float32), trainable=True, name='biases')
cfc1 = tf.Variable(tf.truncated_normal([32, 16]))

#c1
c1_2_conv_ker = tf.nn.conv2d(c1_conv1, conv_k2, [1, 3, 3, 1], padding='SAME')
c1_2_bias = tf.nn.bias_add(c1_2_conv_ker, conv_b2)
c1_2_conv1 = tf.nn.tanh(tf.matmul(tf.reshape(tf.nn.sigmoid(c1_2_bias, name='c2c1'), (-1, 32)),cfc1))


#c2
c2_2_conv_ker = tf.nn.conv2d(c2_conv1, conv_k2, [1, 3, 3, 1], padding='SAME')
c2_2_bias = tf.nn.bias_add(c2_2_conv_ker, conv_b2)
c2_2_conv1 = tf.nn.tanh(tf.matmul(tf.reshape(tf.nn.sigmoid(c2_2_bias, name='c2c2'), (-1, 32)),cfc1))


#output
O = tf.concat([c1_2_conv1, c2_2_conv1],axis=1)
Ow1 = tf.Variable(tf.truncated_normal([1, 16*2], dtype=tf.float32, stddev=1e-1))
Ob1 = tf.Variable(tf.truncated_normal([1], dtype=tf.float32, stddev=1e-1))
out = tf.nn.tanh(tf.matmul(O, tf.transpose(Ow1)) + Ob1)
loss = tf.reduce_sum(tf.nn.l2_loss(out-D))


sess = tf.Session()
opt = tf.train.AdamOptimizer(1e-6)
train = opt.minimize(loss)
sess.run(tf.global_variables_initializer())

for step in range(5000):
    if step % 10 == 0:
        print(sess.run(loss, feed_dict={C1:X1, C2:X2, D:Y}))

    sess.run(train, feed_dict={C1:X1, C2:X2, D:Y})


V = sess.run(c1_2_conv1, feed_dict={C1:X1})


import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

p = PCA(n_components=2)
U = p.fit_transform(V)
plt.scatter(U[:,0],U[:,1])
plt.show()


