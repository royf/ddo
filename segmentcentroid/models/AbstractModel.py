
"""
Defines an abstract model
"""


class AbstractModel(object):

    def __init__(self, statedim, actiondim, discrete=True, unnormalized=False):
        self.statedim = statedim
        self.actiondim = actiondim
        self.discrete = True 
        self.unnormalized = unnormalized

    #returns a probability distribution over actions
    def eval(self,s):
        raise NotImplemented("Must implement an eval function")

    #return the log derivative log \nabla_\theta \pi(s)
    def log_deriv(self, s):
        raise NotImplemented("Must implement an eval function")


    def descent(self, grad_theta, learning_rate):
        raise NotImplemented("Must implement a descent step")

    def visited(self, s):
        return True
