import cityflow
import numpy as np

import math
import random

from itertools import count
import torch
import torch.optim as optim

from collections import namedtuple
import torch.nn as nn
import torch.nn.functional as F
import random

import argparse
import os

from dqn import DQN, ReplayMemory, optimize_model
from environ import Environment
from logger import Logger

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--sim_config", default='../scenarios/4x4/1.config',  type=str, help="the relative path to the simulation config file")

    parser.add_argument("--num_episodes", default=1, type=int,
                        help="the number of episodes to run (one episosde consists of a full simulation run for num_sim_steps)"
                        )
    parser.add_argument("--num_sim_steps", default=1800, type=int, help="the number of simulation steps, one step corresponds to 1 second")
    parser.add_argument("--agents_type", default='analytical', type=str, help="the type of agents learning/policy/analytical/hybrid/demand")

    parser.add_argument("--update_freq", default=10, type=int,
                        help="the frequency of the updates (training pass) of the deep-q-network, default=10")
    parser.add_argument("--batch_size", default=64, type=int, help="the size of the mini-batch used to train the deep-q-network, default=64")
    parser.add_argument("--lr", default=5e-4, type=float, help="the learning rate for the dqn, default=5e-4")
    parser.add_argument("--eps_start", default=1, type=float, help="the epsilon start")
    parser.add_argument("--eps_end", default=0.01, type=float, help="the epsilon decay")
    parser.add_argument("--eps_decay", default=5e-5, type=float, help="the epsilon decay")
    parser.add_argument("--eps_update", default=1799, type=float, help="how frequently epsilon is decayed")
    parser.add_argument("--load", default=None, type=str, help="path to the model to be loaed")
    parser.add_argument("--mode", default='train', type=str, help="mode of the run train/test")
    parser.add_argument("--replay", default=False, type=bool, help="saving replay")
    parser.add_argument("--mfd", default=True, type=bool, help="saving mfd data")
    parser.add_argument("--path", default='../', type=str, help="path to save data")
    parser.add_argument("--meta", default=False, type=bool, help="indicates if meta learning for ML")
    parser.add_argument("--load_cluster", default=None, type=str, help="path to the clusters and models to be loaded")
<<<<<<< HEAD
    parser.add_argument("--ID", default=None, type=int, help="id used for naming")
=======
>>>>>>> f7b34e639ea6bfe8e3c280fe628e8c5abed100a3

    return parser.parse_args()


args = parse_args()
logger = Logger(args)
<<<<<<< HEAD

if args.agents_type == 'denflow':
    n_states = 2
else:
    n_states = 57

environ = Environment(args, n_actions=9, n_states=n_states)
=======
environ = Environment(args, n_actions=9, n_states=57)
>>>>>>> f7b34e639ea6bfe8e3c280fe628e8c5abed100a3
        
num_episodes = args.num_episodes
num_sim_steps = args.num_sim_steps

step = 0
best_time = 999999
best_veh_count = 0
best_reward = -999999
saved_model = None
environ.best_epoch = 0
    
environ.eng.set_save_replay(open=False)
environ.eng.set_random_seed(2)
random.seed(2)
np.random.seed(2)

log_phases = False

for i_episode in range(num_episodes):
    logger.losses = []
    if i_episode == num_episodes-1 and args.replay:
        environ.eng.set_save_replay(open=True)
        print(args.path + "../replay_file.txt")
        environ.eng.set_replay_file(args.path + "../replay_file.txt")

    if args.meta:
<<<<<<< HEAD
        config_path =  args.sim_config.split('/')[0] + '/' + args.sim_config.split('/')[1] + '/5x5_900_100_3k/scenarios/' + str(i_episode) + "/" + str(i_episode) + ".config"
        print(config_path)
=======
        config_path =  args.sim_config.split('/')[0] + '/' + args.sim_config.split('/')[1] + '/4x4_10000vehs/' + str(i_episode) + "/" + str(i_episode) + ".config"
>>>>>>> f7b34e639ea6bfe8e3c280fe628e8c5abed100a3
        environ.eng = cityflow.Engine(config_path, thread_num=8)
        
        
    print("episode ", i_episode)
    done = False

    environ.reset()

    t = 0
    while t < num_sim_steps:
        if t >= num_sim_steps-1: done = True
                            
        environ.step(t, done)   
        t += 1
      
        step = (step+1) % environ.update_freq
        if step == 0 and args.mode == 'train':
            if environ.agents_type == 'cluster':
                all_losses = []
                for cluster in environ.cluster_algo.M[-1]:
                    if len(environ.cluster_models.memory_dict[cluster.ID]) > environ.batch_size:
                        experience = environ.cluster_models.memory_dict[cluster.ID].sample()

<<<<<<< HEAD
                        local_net, target_net, optimizer = environ.cluster_models.model_dict[cluster.ID]
                        all_losses.append(optimize_model(experience, local_net, target_net, optimizer))
                        
                        logger.losses.append(np.mean(all_losses))

            elif environ.agents_type == 'learning' or environ.agents_type == 'hybrid' or environ.agents_type == 'denflow':
                if len(environ.memory)>environ.batch_size:
                    experience = environ.memory.sample()
                    logger.losses.append(optimize_model(experience, environ.local_net, environ.target_net, environ.optimizer))

            elif environ.agents_type == 'presslight':
                if len(environ.memory)>environ.batch_size:
                    experience = environ.memory.sample()
                    logger.losses.append(optimize_model(experience, environ.local_net, environ.target_net, environ.optimizer, tau=1))
=======
                        # for name, param in cluster.local_net.named_parameters():
                        #     if param.requires_grad:
                        #         dp1 = param.data[1][1]
                        #         grad =  list(cluster.local_net.parameters())[0].grad
                        #     break

                        local_net, target_net, optimizer = environ.cluster_models.model_dict[cluster.ID]
                        all_losses.append(optimize_model(experience, local_net, target_net, optimizer))

                        # for name, param in cluster.local_net.named_parameters():
                        #     if param.requires_grad:
                        #         dp2 = param.data[1][1]
                        #     break
                        # print(dp1, dp2)
                        logger.losses.append(np.mean(all_losses))

        elif environ.agents_type == 'learning' or environ.agents_type == 'hybrid':
            if len(environ.memory)>environ.batch_size:
                experience = environ.memory.sample()

        elif environ.agents_type == 'presslight':
            if len(environ.memory)>environ.batch_size:
                experience = environ.memory.sample()
                logger.losses.append(optimize_model(experience, environ.local_net, environ.target_net, environ.optimizer, tau=1))
>>>>>>> f7b34e639ea6bfe8e3c280fe628e8c5abed100a3

    if environ.agents_type == 'learning' or environ.agents_type == 'hybrid' or  environ.agents_type == 'presslight':
        if environ.eng.get_average_travel_time() < best_time:
            best_time = environ.eng.get_average_travel_time()
            logger.save_models(environ, flag=False)
            environ.best_epoch = i_episode
                
        if environ.eng.get_finished_vehicle_count() > best_veh_count:
            best_veh_count = environ.eng.get_finished_vehicle_count()
            logger.save_models(environ, flag=True)
            environ.best_epoch = i_episode
<<<<<<< HEAD
            
    elif environ.agents_type == 'cluster':
        if environ.eng.get_average_travel_time() < best_time:
            best_time = environ.eng.get_average_travel_time()
            logger.save_clusters(environ)
            environ.best_epoch = i_episode
=======
>>>>>>> f7b34e639ea6bfe8e3c280fe628e8c5abed100a3
    
    logger.log_measures(environ)

    if environ.agents_type == 'learning' or environ.agents_type == 'hybrid' or  environ.agents_type == 'presslight':
        # if logger.reward > best_reward:
        best_reward = logger.reward
        logger.save_models(environ, flag=None)
    
    print(logger.reward, environ.eng.get_average_travel_time(), environ.eng.get_finished_vehicle_count())

    if environ.agents_type == 'cluster':
        print(len(environ.cluster_algo.M[-1]))
<<<<<<< HEAD
        if len([len(x) for x in environ.cluster_models.memory_dict.values()]) < 10:
            print([len(x) for x in environ.cluster_models.memory_dict.values()])

if environ.agents_type != 'cluster':
    logger.save_models(environ, flag=None)
        
logger.save_log_file(environ)
logger.serialise_data(environ)

=======
        if len([len(x.data) for x in environ.cluster_algo.M[-1]]) < 10:
            print([len(x.data) for x in environ.cluster_algo.M[-1]])
        # print([len(x.data) for i in range(0, 20) for x in environ.cluster_algo.M[i]])

if environ.agents_type == 'cluster':
    logger.save_clusters(environ)

logger.save_models(environ, flag=None)
logger.save_log_file(environ)
logger.serialise_data(environ)

# if environ.agents_type == 'learning' or environ.agents_type == 'hybrid' or environ.agents_type == 'presslight' or environ.agents_type == 'policy':
#     logger.save_measures_plots()



>>>>>>> f7b34e639ea6bfe8e3c280fe628e8c5abed100a3

