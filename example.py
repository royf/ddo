#!/usr/bin/env python

from segmentcentroid.envs.GridWorldEnv import GridWorldEnv
import numpy as np


##Example 1##
#Initialize GridWorld and query a state
MAP_NAME = 'resources/GridWorldMaps/4x5.txt'
gmap = np.loadtxt(MAP_NAME, dtype=np.uint8)
g = GridWorldEnv(gmap)
g.init()
print g.getState()
g.play(3)
print g.getState()

##Example 1##