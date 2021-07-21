import cityflow
import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import random

import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam

from dqn import DQN, ReplayMemory, optimize_model
from learning_agent import Learning_Agent
from analytical_agent import Analytical_Agent
from demand_agent import Demand_Agent
from hybrid_agent import Hybrid_Agent
from presslight_agent import Presslight_Agent
from fixed_agent import Fixed_Agent

from policy_agent import DPGN, Policy_Agent

class Environment:
    """
    The class Environment represents the environment in which the agents operate in this case it is a city
    consisting of roads, lanes and intersections which are controled by the agents
    """
    def __init__(self, args, ID, n_actions=9, n_states=44):
        """
        initialises the environment with the arguments parsed from the user input
        :param args: the arguments input by the user
        :param n_actions: the number of possible actions for the learning agent, corresponds to the number of available phases
        :param n_states: the size of the state space for the learning agent
        """
        self.eng = cityflow.Engine(args.sim_config, thread_num=8)
        self.ID = ID
        
        self.update_freq = args.update_freq      # how often to update the network
        self.batch_size = args.batch_size
        
        self.eps_start = args.eps_start
        self.eps_end = args.eps_end
        self.eps_decay= args.eps_decay
        self.eps_update = args.eps_update
        
        self.eps = self.eps_start

        self.agents = []

        random.seed(2)

        self.agents_type = args.agents_type

        agent_ids = [x for x in self.eng.get_intersection_ids() if not self.eng.is_intersection_virtual(x)]

        self.agents = []

        self.action_freq = 10   #typical update freq for agents

        
        # self.n_actions = len(self.agents[0].phases)
        self.n_states = n_states

        # if args.load:
        #     self.local_net = DQN(n_states, self.n_actions, seed=2).to(self.device)
        #     self.local_net.load_state_dict(torch.load(args.load))
        #     self.local_net.eval()
            
        #     self.target_net = DQN(n_states, self.n_actions, seed=2).to(self.device)
        #     self.target_net.load_state_dict(torch.load(args.load))
        #     self.target_net.eval()
        # else:
        #     self.local_net = DQN(n_states, self.n_actions, seed=2).to(self.device)
        #     self.target_net = DQN(n_states, self.n_actions, seed=2).to(self.device)

        # self.optimizer = optim.Adam(self.local_net.parameters(), lr=args.lr, amsgrad=True)
        # self.memory = ReplayMemory(self.n_actions, batch_size=args.batch_size)

        
    def step(self, time, done):
        """
        represents a single step of the simulation for the analytical agent
        :param time: the current timestep
        :param done: flag indicating weather this has been the last step of the episode, used for learning, here for interchangability of the two steps
        """
        # print(time)
        lane_vehs = self.eng.get_lane_vehicles()
        lanes_count = self.eng.get_lane_vehicle_count()

        veh_distance = 0
        if self.agents_type == "hybrid" or self.agents_type == "learning":
            veh_distance = self.eng.get_vehicle_distance()

        for agent in self.agents:
            if agent.agents_type == "hybrid":
                agent.update_arr_dep_veh_num(lane_vehs)
            agent.step(self.eng, time, lane_vehs, lanes_count, veh_distance, self.eps, done)

        # if time % self.action_freq == 0: self.eps = max(self.eps-self.eps_decay,self.eps_end)
        if time % self.eps_update == 0: self.eps = max(self.eps*self.eps_decay,self.eps_end)

        self.eng.next_step()

    def reset(self):
        """
        resets the movements amd rewards for each agent and the simulation environment, should be called after each episode
        """
        self.eng.reset(seed=False)

        for agent in self.agents:
            agent.reset_movements()
            agent.total_rewards = 0
            agent.reward_count = 0
            agent.action_type = 'act'



