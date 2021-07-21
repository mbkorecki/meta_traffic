import cityflow
import numpy as np
import torch

import math
import random
from itertools import product

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



def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--path", default='../scenarios/1x1/',  type=str, help="the relative path to the simulation config file")

    parser.add_argument("--num_episodes", default=1, type=int,
                        help="the number of episodes to run (one episosde consists of a full simulation run for num_sim_steps)")
    parser.add_argument("--num_sim_steps", default=300, type=int, help="the number of simulation steps, one step corresponds to 1 second")
    parser.add_argument("--agents_type", default='learning', type=str, help="the type of agents learning/policy/analytical/hybrid/demand")

    parser.add_argument("--update_freq", default=10, type=int,
                        help="the frequency of the updates (training pass) of the deep-q-network, default=10")
    parser.add_argument("--batch_size", default=64, type=int, help="the size of the mini-batch used to train the deep-q-network, default=64")
    parser.add_argument("--lr", default=5e-4, type=float, help="the learning rate for the dqn, default=5e-4")
    parser.add_argument("--eps_start", default=1, type=float, help="the epsilon start")
    parser.add_argument("--eps_end", default=0.01, type=float, help="the epsilon decay")
    parser.add_argument("--eps_decay", default=0.95, type=float, help="the epsilon decay")
    # parser.add_argument("--eps_decay", default=5e-5, type=float, help="the epsilon decay")
    parser.add_argument("--eps_update", default=1799, type=float, help="how frequently epsilon is decayed")
    parser.add_argument("--load", default=None, type=str, help="path to the model to be loaded")

    return parser.parse_args()


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

def train(agent, environ, args, test=False):
    logger = Logger(args)
    step = 0
    environ.agents = [agent]
    for i_episode in range(args.num_episodes):
        done = False
        environ.reset()
        t = 0

        while t < args.num_sim_steps:
            if t >= args.num_sim_steps-1: done = True
                            
            environ.step(t, done)   
            t += 1
      
            step = (step+1) % environ.update_freq
            if test == False and (agent.agents_type == 'learning' or agent.agents_type == 'hybrid') and step == 0:
                for agent in environ.agents:
                    if len(agent.memory)>environ.batch_size:
                        experience = agent.memory.sample()
                        logger.losses.append(optimize_model(experience, agent.local_net, agent.target_net, agent.optimizer))

        if agent.agents_type == 'analytical':
            break
        
    logger.log_measures(environ)
    return logger.reward, environ.eng.get_average_travel_time(), environ.eng.get_finished_vehicle_count()

def generate_baseline(pop_envs):
    total_freeflow_throughput = []
    for env in pop_envs:
        agent_ids = [x for x in env.eng.get_intersection_ids() if not env.eng.is_intersection_virtual(x)]
        agent_id = agent_ids[0]
        baseline_agent = Analytical_Agent(env.eng, ID=agent_id)
        env.analytic_reward, env.analytic_avg_travel_time, env.analytic_throughput = train(baseline_agent, env, args)
        total_freeflow_throughput += [env.freeflow_throughput]
        
    analytic_reward = np.mean([env.analytic_reward for env in pop_envs])
    analytic_reward_std = np.std([env.analytic_reward for env in pop_envs])
    analytic_avg_travel_time = np.mean([env.analytic_avg_travel_time for env in pop_envs])
    analytic_throughput = np.mean([env.analytic_throughput for env in pop_envs])

    return analytic_reward, analytic_reward_std, analytic_avg_travel_time, analytic_throughput, total_freeflow_throughput


def meta_train_loop(meta_train_episodes, pop_agents, pop_envs, args):
    analytic_reward, analytic_reward_std, analytic_avg_travel_time, analytic_throughput, freeflow_throughput = generate_baseline(pop_envs)
    for i in range(meta_train_episodes):
        print("META EPISODE", i)
        results = []
        for agent, env in zip(pop_agents, pop_envs):
            reward, avg_travel_time, throughput = train(agent, env, args)
            results.append((reward, avg_travel_time, throughput))
        learning_results = results
        
        print(np.mean([x[0] for x in learning_results]), np.std([x[0] for x in learning_results]), np.mean([x[1] for x in learning_results]), np.mean([x[2] for x in learning_results]))
        print(analytic_reward, analytic_reward_std, analytic_avg_travel_time, analytic_throughput, np.mean(freeflow_throughput), np.std(freeflow_throughput))

        random.shuffle(pop_agents)
        random.shuffle(pop_envs)


def evolve_agents(agents, environs, args):
    for agent in agents:
        rewards = []
        avg_travel_times = throughputs = 0
        for env in environs:
            reward, avg_travel_time, throughput = train(agent, env, args, test=True)
            rewards.append(reward)

        agent.fitness = np.mean(rewards)

    agents.sort(key=lambda x: x.fitness, reverse=False)
    print([x.fitness for x in agents])
    pop_parents = agents[0:int(len(environs)/2)]
    new_population = []
    
    eng = cityflow.Engine(args.path + "0.config", thread_num=8)
    agent_ids = [x for x in eng.get_intersection_ids() if not eng.is_intersection_virtual(x)]
    agent_id = agent_ids[0]

    while len(new_population) != len(agents)-1:    
        parents = random.sample(pop_parents, 2)
        child = Hybrid_Agent(eng, ID=agent_id, in_roads=eng.get_intersection_in_roads(agent_id),
                             out_roads=eng.get_intersection_out_roads(agent_id), n_states=args.n_states, lr=args.lr, batch_size=args.batch_size)
        
        for param1, param2 in zip(parents[0].local_net.named_parameters(), parents[1].local_net.named_parameters()):
            name1 = param1[0]
            name2 = param2[0]
            param1 = param1[1]
            param2 = param2[1]

            dim = param1.shape[0]
            crossover_point = random.randrange(0, dim)
            new_params = torch.cat((param1[0:crossover_point], param2[crossover_point:]))
            
            if random.random() <= 0.1:
                #mutation
                mutation_point = random.randrange(0, dim)
                new_params[mutation_point] += np.random.normal(0, 0.2)

            child_params = dict(child.local_net.named_parameters())
            child_params[name1].data.copy_(new_params)

        new_population.append(child)

    child = Hybrid_Agent(eng, ID=agent_id, in_roads=eng.get_intersection_in_roads(agent_id),
                         out_roads=eng.get_intersection_out_roads(agent_id), n_states=args.n_states, lr=args.lr, batch_size=args.batch_size)
    child_params = dict(child.local_net.named_parameters())
        
    for parent in pop_parents:
        for name, param in parent.local_net.named_parameters():
            child_params[name].data.copy_(child_params[name].data + param)

    for name, _ in child.local_net.named_parameters():
        child_params[name].data.copy_(child_params[name] / len(parents))
    new_population.append(child)
        
    return new_population             

def evolve_environs(environs):
    for env in environs:
        env.fitness = env.analytic_throughput / env.freeflow_throughput
        
    environs.sort(key=lambda x: x.fitness, reverse=False)
    pop_parents = environs[0:int(len(environs))]

    genotypes = []
    new_population = []

    while len(new_population) != len(environs):    
        parents = random.sample(pop_parents, 2)
        crossover_point = random.randrange(0, 11)
        new_genotype = parents[0].genotype[0:crossover_point] + parents[1].genotype[crossover_point:]
        if random.random() <= 0.1:
            #mutation
            mutation_point = random.randrange(0, 11)
            if random.random() > 0.5:
                new_genotype[mutation_point] *= 1.5
            else:
                new_genotype[mutation_point] /= 1.5
        new_population.append(new_genotype)
        
    return new_population

        
if __name__ == '__main__':
    np.random.seed(2)
    random.seed(2)
    args = parse_args()

    pop_agent_size = 10
    pop_envs_size = 10

    genotypes = []
    for _ in range(pop_envs_size):
        genotypes.append(list(np.abs(np.random.normal(0.05, 0.05, size=12))))
    
    pop_envs = populate_envs(pop_envs_size, args, genotypes)
    pop_agents = populate_agents(pop_agent_size, args)
    
    meta_train_episodes = 5
    epochs = 100
    
    for epoch in range(epochs):
        print(epoch)
        meta_train_loop(meta_train_episodes, pop_agents, pop_envs, args)
        # genotypes = evolve_environs(pop_envs)
        # pop_envs = populate_envs(pop_envs_size, args, genotypes)
        pop_agents = evolve_agents(pop_agents, pop_envs, args)
