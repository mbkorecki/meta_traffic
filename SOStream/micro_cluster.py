from src.dqn import DQN, ReplayMemory

import torch
import torch.optim as optim
import copy

glob_n_states = 57

class MicroCluster:

    def __init__(self, centroid, data=[], number_points = 1, radius = 0, memory=None, local_net=None, target_net=None, optimizer=None):
        self.number_points = number_points
        self.radius = radius
        self.centroid = centroid

        self.data = data

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        #merging of memory etc.
        if memory:
            self.memory = memory
        else:
            self.memory = ReplayMemory(action_size=9, batch_size=64)

        #define functions for initialising new dqns from neighbours etc.
        if local_net:
            self.local_net = local_net
        else:
            self.local_net = DQN(glob_n_states, 9, seed=2).to(self.device)

        if target_net:
            self.target_net = target_net
        else:
            self.target_net = DQN(glob_n_states, 9, seed=2).to(self.device)

        if optimizer:
            self.optimizer = optimizer
        else:
            self.optimizer = optim.Adam(self.local_net.parameters(), lr=5e-4, amsgrad=True)
        
    pass
