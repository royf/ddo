
from AbstractEnvironment import *

"""
This class defines an abstract environment,
all environments derive from this class
"""

class GridWorldEnv(AbstractEnvironment):


    ##All of the constant variables

    # Constants in the map
    EMPTY, BLOCKED, START, GOAL, PIT, AGENT = range(6) #codes

    ACTIONS = np.array([[-1, 0], [+1, 0], [0, -1], [0, +1]])
    actions_num = 4
    GOAL_REWARD = +1
    PIT_REWARD = -1
    STEP_REWARD = -.001

    #takes in a 2d integer map coded by the first line of comments
    def __init__(self, gmap, noise=0.1):

        self.map = gmap
        self.start_state = np.argwhere(self.map == self.START)[0]
        self.ROWS, self.COLS = np.shape(self.map)
        self.statespace_limits = np.array(
            [[0, self.ROWS - 1], [0, self.COLS - 1]])
        self.NOISE = noise

        super(GridWorldEnv, self).__init__()


    #helper method returns the terminal state
    def isTerminal(self, s=None):
        if s is None:
            s = self.state
        if self.map[s[0], s[1]] == self.GOAL:
            return True
        if self.map[s[0], s[1]] == self.PIT:
            return True
        return False


    #helper method returns the possible actions
    def possibleActions(self, s=None):
        if s is None:
            s = self.state
        possibleA = np.array([], np.uint8)
        for a in xrange(self.actions_num):
            ns = s + self.ACTIONS[a]
            if (
                    ns[0] < 0 or ns[0] == self.ROWS or
                    ns[1] < 0 or ns[1] == self.COLS or
                    self.map[int(ns[0]), int(ns[1])] == self.BLOCKED):
                continue
            possibleA = np.append(possibleA, [a])
        return possibleA


    """
    This function initializes the envioronment
    """
    def init(self, state=None, time=0, reward=0 ):
        if state == None:
            self.state = self.start_state.copy()
        else:
            self.state = state

        self.time = time
        self.reward = reward
        self.termination = self.isTerminal()


    """
    This function returns the current state, time, total reward, and termination
    """
    def getState(self):
        return self.state, self.time, self.reward, self.termination


    """
    This function takes an action
    """
    def play(self, action):
        r = self.STEP_REWARD
        ns = self.state.copy()
        if np.random.rand(1,1) < self.NOISE:
            # Random Move
            a = np.random.choice(self.possibleActions())

        # Take action
        ns = self.state + self.ACTIONS[a]

        # Check bounds on state values
        if (ns[0] < 0 or ns[0] == self.ROWS or
            ns[1] < 0 or ns[1] == self.COLS or
            self.map[ns[0], ns[1]] == self.BLOCKED):
            ns = self.state.copy()
        else:
            # If in bounds, update the current state
            self.state = ns.copy()

        # Compute the reward
        if self.map[ns[0], ns[1]] == self.GOAL:
            r = self.GOAL_REWARD
        if self.map[ns[0], ns[1]] == self.PIT:
            r = self.PIT_REWARD

        self.state = ns
        self.time = time + 1
        self.reward = reward + r
        self.termination = self.isTerminal()


    """
    This function rolls out a policy which is a map from state to action
    """
    def rollout(self, policy):
        trajectory = []

        while not self.terminated:
            self.play(policy(self.state))
            trajectory.append(self.getState())

        return trajectory


        

     
