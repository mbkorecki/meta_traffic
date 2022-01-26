import random
import numpy as np
import torch

from hybrid_agent import Hybrid_Agent


class Cluster_Agent(Hybrid_Agent):

    def __init__(self, eng, ID='', in_roads=[], out_roads=[], n_states=None, lr=None, batch_size=None):
        super().__init__(eng, ID, in_roads, out_roads, n_states, lr, batch_size)
        self.agents_type = 'cluster'
        self.assigned_cluster = None
            
    def step(self, eng, time, lane_vehs, lanes_count, veh_distance, eps, clustering, done):
        if time % self.action_freq == 0:
            if self.action_type == "reward":
                reward = self.get_reward(lanes_count)
                self.reward = reward
                self.total_rewards += [reward]
                self.reward_count += 1
                reward = torch.tensor([reward], dtype=torch.float)
                next_state = torch.FloatTensor(self.observe(eng, time, lanes_count, lane_vehs, veh_distance)).unsqueeze(0)

                self.assigned_cluster.memory.add(self.state, self.action.ID, reward, next_state, done)
                self.action_type = "act"

            if self.action_type == "act":
                self.state = np.asarray(self.observe(eng, time, lanes_count, lane_vehs, veh_distance))

                # self.assigned_cluster = clustering.process(self.state)
                # self.assigned_cluster = clustering.process(np.asarray([self.get_reward(lanes_count), self.phase.ID]))
                self.assigned_cluster = clustering.process(self.get_reward(lanes_count))
                # self.assigned_cluster = clustering.process(np.asarray([self.get_reward(lanes_count), time]))


                self.action = self.act(self.assigned_cluster.local_net, self.state, time, lanes_count, eps=eps)
                self.green_time = 10

                if self.action != self.phase:
                    self.update_wait_time(time, self.action, self.phase, lanes_count)
                    self.set_phase(eng, self.clearing_phase)
                    self.action_type = "update"
                    self.action_freq = time + self.clearing_time
                    
                else:
                    self.action_type = "reward"
                    self.action_freq = time + self.green_time

            elif self.action_type == "update":
                self.set_phase(eng, self.action)
                self.action_type = "reward"
                self.action_freq = time + self.green_time
