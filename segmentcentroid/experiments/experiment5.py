#!/usr/bin/env python

from segmentcentroid.envs.GridWorldEnv import GridWorldEnv
from segmentcentroid.tfmodel.GridWorldModel import GridWorldModel
from segmentcentroid.planner.jigsaws_loader import JigsawsPlanner
from segmentcentroid.tfmodel.JHUJigSawsModel import JHUJigSawsModel
from segmentcentroid.planner.traj_utils import *

import numpy as np
import copy

import tensorflow as tf


def runPolicies(demonstrations=20,
                directory='/Users/sanjayk/Downloads/Knot_Tying/kinematics/AllGestures/',
                pretrain=1000,
                super_iterations=100,
                sub_iterations=100,
                learning_rate=1e-3):

    j = JigsawsPlanner(directory)

    full_traj = []
    for i in range(0,demonstrations):
        full_traj.append(j.plan())
    
    m  = JHUJigSawsModel(8)

    opt = tf.train.AdamOptimizer(learning_rate=learning_rate)

    m.pretrain(opt, full_traj, pretrain)
    
    m.train(opt, full_traj, super_iterations, sub_iterations)

    j.visualizePlans(full_traj, m, filename="resources/results/exp5-trajs2.png")




