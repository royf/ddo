"""
This class defines a loader that loads 
the JHU Jigsaws dataset
"""
from os import listdir
from os.path import isfile, join
import numpy as np
from .AbstractPlanner import *

class JigsawsPlanner(AbstractPlanner):

    def __init__(self, directory, arms=['left', 'right']):

        #declaring variables that should be set
        self.directory = directory
        self.arms = arms

        if arms == ['left', 'right']:
            self.START = 39
            self.END = 76

        elif 'left' in arms:
            self.START = 39
            self.END = 58

        elif 'right' in arms:
            self.START = 58
            self.END = 76

        else:

            raise ValueError("Invalid Arm Config")


        super(JigsawsPlanner, self).__init__(None)


    """
    This function returns a trajectory [(s,a) tuples]
    """
    def plan(self, max_depth=-1, start=None):
        files = [f for f in listdir(self.directory) if isfile(join(self.directory, f))]
        index = np.random.choice(len(files))
        f = open(self.directory+"/"+files[index], "r")
        
        lines = f.readlines()
        states = [np.array([float(li) for li in l.split()][self.START:self.END]) for l in lines]

        if max_depth == -1:
            max_depth = len(states)
        else:
            max_depth = min(max_depth, len(states))

        traj = []

        for t in range(1, max_depth):
            xt = states[t-1]
            xtp = states[t]
            traj.append((xt, xtp-xt))

        return traj


    def visualizePlan(self, traj, segmentation=None):
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        colors = [(1,0,0,1), (0,1,0,1), (0,0,1,1), (1,0,1,1), (1,0,0,1), (1,1,1,1), (0,0,0,1), (0.5,0.5,0.5,1)]

        if segmentation == None:
            segmentation = np.ravel(np.zeros((len(traj),1)))

        segkeys = set(segmentation)

        segset = {}

        for i,s in enumerate(segmentation):
            if s not in segset:
                segset[s] = set()

            segset[s].add(i)


        fig = plt.figure()

        if len(self.arms) == 2:
            ax = fig.add_subplot(121, projection='3d')

            for k in segkeys:
                scatCollection = ax.scatter([t[0][0] for i,t in enumerate(traj) if i in segset[k]], 
                                            [t[0][1] for i,t in enumerate(traj) if i in segset[k]], 
                                            [t[0][2] for i,t in enumerate(traj) if i in segset[k]], 
                                            color=colors[k])

            ax = fig.add_subplot(122, projection='3d')

            for k in segkeys:
                ax.scatter([t[0][19] for i,t in enumerate(traj) if i in segset[k]], 
                            [t[0][20] for i,t in enumerate(traj) if i in segset[k]], 
                            [t[0][21] for  i,t in enumerate(traj) if i in segset[k]], 
                            color=colors[k])
            
            plt.show()
        else:
            ax = fig.add_subplot(111, projection='3d')

            for k in segkeys:
                ax.scatter([t[0][0] for i,t in enumerate(traj) if i in segset[k]], 
                           [t[0][1] for i,t in enumerate(traj) if i in segset[k]], 
                           [t[0][2] for i,t in enumerate(traj) if i in segset[k]], 
                           color=colors[k])
            
            plt.show()





        

