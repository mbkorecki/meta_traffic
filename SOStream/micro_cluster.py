from src.dqn import DQN, ReplayMemory

import torch
import torch.optim as optim
import copy

glob_n_states = 57

class MicroCluster:

    def __init__(self, centroid, data=[], number_points = 1, radius = 0, ID=None):
        self.number_points = number_points
        self.radius = radius
        self.centroid = centroid

        self.ID = ID
        self.data = data

    pass
