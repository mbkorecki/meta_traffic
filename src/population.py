import cityflow
import numpy as np
import torch

import math
import random
import itertools

import argparse
import os
import json

from dqn import DQN, ReplayMemory, optimize_model
from environ import Environment
from logger import Logger

from learning_agent import Learning_Agent
from analytical_agent import Analytical_Agent
from hybrid_agent import Hybrid_Agent


class Env_Config:
    
    def __init__(self, ID, path, genotype=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]):
        """
        initialises the evolving environment
        :param ID: the unique ID of the environment
        :param genotype: specifies the flow dynamics of the environ, a list of 12 elements, each indicating a flow at a given movement
        """
        self.ID = ID
        self.genotype = genotype
        self.path = path
        self.config = self.path + str(self.ID) + ".config"

    def add_vehicle(self, data, route, interval, start_time):
        vehicle_dict = {"vehicle": {"length": 5.0, "width": 2.0, "maxPosAcc": 2.0, "maxNegAcc": 4.5, "usualPosAcc": 2.0,
                                    "usualNegAcc": 4.5, "minGap": 2.5, "maxSpeed": 11.11, "headwayTime": 1.5},
                        "route": route, "interval": interval, "startTime": start_time, "endTime": -1}
        data.append(vehicle_dict)
        
    def generate_flow_file(self):
        data = []

        for flow_rate, movement in zip(self.genotype, movements.values()):
            if 1/flow_rate >= 1:
                stat_time = random.randrange(0, 10)
                self.add_vehicle(data, movement, int(1/flow_rate), stat_time)

        with open(self.path + "flow" + str(self.ID) + ".json", "w") as write_file:
            json.dump(data, write_file)


    def generate_config_file(self):
        data = {"interval": 1, "seed": 0, "dir": "../scenarios/1x1/", "roadnetFile": "roadnet.json",
                    "flowFile": "flow" + str(self.ID) + ".json", "rlTrafficLight": True, "saveReplay": False, "roadnetLogFile":
                    "roadnetLogFile.json", "replayLogFile": "replayLogFile.txt", "laneChange" : True}

        with open(self.path + str(self.ID) + ".config", "w") as write_file:
            json.dump(data, write_file)
            
movements = {
    'WE' : ["road_0_1_0", "road_1_1_0"],
    'WN' : ["road_0_1_0", "road_1_1_1"],
    'WS' : ["road_0_1_0", "road_1_1_3"],
    'SE' : ["road_1_0_1", "road_1_1_0"],
    'SN' : ["road_1_0_1", "road_1_1_1"],
    'SW' : ["road_1_0_1", "road_1_1_2"],
    'EN' : ["road_2_1_2", "road_1_1_1"],
    'EW' : ["road_2_1_2", "road_1_1_2"],
    'ES' : ["road_2_1_2", "road_1_1_3"],
    'NE' : ["road_1_2_3", "road_1_1_0"],
    'NW' : ["road_1_2_3", "road_1_1_2"],
    'NS' : ["road_1_2_3", "road_1_1_3"]
}



def populate_envs(size, args, genotypes):
    population_envs = []
    for ID, genotype in zip(range(size), genotypes):
        env_config = Env_Config(ID, args.path, genotype)
        env_config.generate_flow_file()
        env_config.generate_config_file()

        args.sim_config = args.path + str(ID) + ".config"
        environ = Environment(args, ID, n_actions=9, n_states=57)
        environ.genotype = env_config.genotype
        environ.freeflow_throughput = np.sum(env_config.genotype * args.num_sim_steps)
                
        population_envs.append(environ)
    
    return population_envs

def populate_agents(size, args):
    #this eng is only for the roadnet data needed to init agents
    eng = cityflow.Engine(args.path + "0.config", thread_num=8)
    agent_ids = [x for x in eng.get_intersection_ids() if not eng.is_intersection_virtual(x)]
    agent_id = agent_ids[0]
    n_actions = 9
    n_states = 57
    args.n_states = n_states
    
    agents = []
    
    for _ in range(size):
        if args.agents_type == 'analytical':
            new_agent = Analytical_Agent(eng, ID=agent_id)
        elif args.agents_type == 'hybrid':
            new_agent = Hybrid_Agent(eng, ID=agent_id, in_roads=eng.get_intersection_in_roads(agent_id),
                                     out_roads=eng.get_intersection_out_roads(agent_id), n_states=n_states, lr=args.lr, batch_size=args.batch_size)
        elif args.agents_type == 'learning':
            new_agent = Learning_Agent(eng, ID=agent_id, in_roads=eng.get_intersection_in_roads(agent_id),
                                       out_roads=eng.get_intersection_out_roads(agent_id), n_states=n_states, lr=args.lr, batch_size=args.batch_size)
        else:
            raise Exception("The specified agent type:", args.agents_type, "is incorrect, choose from: analytical/learning/hybrid")  

        if len(new_agent.phases) <= 1:
            new_agent.set_phase(eng, new_agent.phases[0])
        else:
            agents.append(new_agent)

    return agents

