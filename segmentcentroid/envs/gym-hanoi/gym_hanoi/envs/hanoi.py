import sys

import gym
import numpy as np


class Hanoi(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, n_discs, n_pegs=3):
        self.n_discs = n_discs
        self.n_pegs = n_pegs
        self._reset()

    def _step(self, action):
        fr, to = action
        if fr in self.state:
            fr_disc = self.state.index(fr)
            if (to not in self.state) or (fr_disc < self.state.index(to)):
                self.state[fr_disc] = to
        observation = [disc_peg == peg for peg in range(self.n_pegs) for disc_peg in self.state]
        done = all(peg == 0 for peg in self.state)
        reward = 0 if done else -1
        return observation, reward, done, {}

    def _reset(self):
        self.state = list(np.random.randint(0, self.n_pegs, self.n_discs))

    def _render(self, mode='human', close=False):
        outfile = sys.stdout
        discs = [[i for i, disc_peg in enumerate(self.state) if disc_peg == peg] for peg in range(self.n_pegs)]
        while any(discs):
            m = min(len(d) for d in discs)
            for peg in range(self.n_pegs):
                if len(discs[peg]) == m:
                    outfile.write(' %d ' % discs[peg].pop(0))
                else:
                    outfile.write(' | ')
            outfile.write('\n')
        outfile.write('=' * (3 * self.n_pegs))
        outfile.write('\n')


class Hanoi5Discs(Hanoi):
    def __init__(self):
        super().__init__(5)


class Hanoi10Discs(Hanoi):
    def __init__(self):
        super().__init__(10)
