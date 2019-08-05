import numpy as np


class NormalDistribution(object):
    """
    The normal distribution.
    """
    def __init__(self):
        self.mu = 0
        self.sigma = 1

    def sample(self, N):
        samples = np.random.normal(self.mu, self.sigma, N)
        return samples
