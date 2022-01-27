from SOStream.sostream import SOStream
from dqn import DQN, ReplayMemory
import random
import numpy as np

import torch
import torch.optim as optim


class Cluster_Models():
    
    def __init__(self, n_states, n_actions, lr, batch_size):
        
        self.n_states = n_states
        self.n_actions = n_actions
        self.lr = lr
        self.batch_size = batch_size

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model_dict = {}
        self.memory_dict = {}

    def merge_models(self, a_ID, b_ID):
        """
        merges two models, a_ID becomes the ID of the new model
        :param a_ID: ID of the model to be merged
        :param b_ID: ID of the other model to be merged
        """
        if len(self.memory_dict[a_ID]) + len(self.memory_dict[b_ID]) > 0:
            weight_a = len(self.memory_dict[a_ID]) / (len(self.memory_dict[a_ID]) + len(self.memory_dict[b_ID]))
            weight_b = len(self.memory_dict[b_ID]) / (len(self.memory_dict[a_ID]) + len(self.memory_dict[b_ID]))
        else:
            weight_a = 1
            weight_b = 1

        local_net_a, target_net_a, _ = self.model_dict[a_ID]
        local_net_b, target_net_b, _ = self.model_dict[b_ID]

        local_sd_a = local_net_a.state_dict()
        local_sd_b = local_net_b.state_dict()
        target_sd_a = target_net_a.state_dict()
        target_sd_b = target_net_b.state_dict()

        for key in local_sd_a:
            local_sd_b[key] = weight_a * local_sd_a[key]  + weight_b * local_sd_b[key]
            target_sd_b[key] = weight_a * target_sd_a[key]  + weight_b * local_sd_b[key]

        if len(self.memory_dict[a_ID]) > len(self.memory_dict[b_ID]):
            self.memory_dict[a_ID] += self.memory_dict[b_ID]
            self.model_dict[a_ID][0].load_state_dict(local_sd_b)
            self.model_dict[a_ID][1].load_state_dict(target_sd_b)

            self.model_dict[b_ID] = self.model_dict[a_ID]
            self.memory_dict[b_ID] = self.memory_dict[a_ID]
        else:
            self.memory_dict[b_ID] += self.memory_dict[a_ID]
            self.model_dict[b_ID][0].load_state_dict(local_sd_b)
            self.model_dict[b_ID][1].load_state_dict(target_sd_b)

            self.model_dict[a_ID] = self.model_dict[b_ID]
            self.memory_dict[a_ID] = self.memory_dict[b_ID]
            
    def add_model(self, ID):

        if ID not in self.model_dict.keys():
        
            local_net = DQN(self.n_states, self.n_actions, seed=2).to(self.device)
            target_net = DQN(self.n_states, self.n_actions, seed=2).to(self.device)
            optimizer = optim.Adam(local_net.parameters(), lr=self.lr, amsgrad=True)

            memory = ReplayMemory(self.n_actions, buffer_size=int(1e5), batch_size=self.batch_size)
            
            self.model_dict.update({ID : (local_net, target_net, optimizer)})
            self.memory_dict.update({ID : memory})


    def delete_model(self, ID):
        del self.memory_dict[ID]
        del self.model_dict[ID]
