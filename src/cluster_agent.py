import random
import numpy as np
import torch
<<<<<<< HEAD
import queue
import operator
=======
>>>>>>> f7b34e639ea6bfe8e3c280fe628e8c5abed100a3

from hybrid_agent import Hybrid_Agent


class Cluster_Agent(Hybrid_Agent):

    def __init__(self, eng, ID='', in_roads=[], out_roads=[], n_states=None, lr=None, batch_size=None):
        super().__init__(eng, ID, in_roads, out_roads, n_states, lr, batch_size)
        self.agents_type = 'cluster'
        self.assigned_cluster = None
<<<<<<< HEAD
        self.action_queue = queue.Queue()
            
    def step(self, eng, time, lane_vehs, lanes_count, veh_distance, eps, cluster_algo, cluster_models, done):
        self.update_arr_dep_veh_num(lane_vehs, lanes_count)
=======
            
    def step(self, eng, time, lane_vehs, lanes_count, veh_distance, eps, cluster_algo, cluster_models, done):
>>>>>>> f7b34e639ea6bfe8e3c280fe628e8c5abed100a3
        if time % self.action_freq == 0:
            if self.action_type == "reward":
                reward = self.get_reward(lanes_count)
                self.reward = reward
                self.total_rewards += [reward]
                self.reward_count += 1
                reward = torch.tensor([reward], dtype=torch.float)
                next_state = torch.FloatTensor(self.observe(eng, time, lanes_count, lane_vehs, veh_distance)).unsqueeze(0)

                cluster_models.memory_dict[self.assigned_cluster_id].add(self.state, self.action.ID, reward, next_state, done)
                self.action_type = "act"

            if self.action_type == "act":
                self.state = np.asarray(self.observe(eng, time, lanes_count, lane_vehs, veh_distance))

                # self.assigned_cluster_id = cluster_algo.process(self.state, cluster_models)
                # self.assigned_cluster_id = cluster_algo.process(np.asarray([self.get_reward(lanes_count), self.phase.ID]), cluster_models)
                # self.assigned_cluster_id = cluster_algo.process(np.asarray([self.get_reward(lanes_count), time]), cluster_models)

<<<<<<< HEAD
                flow_change, density_change = self.get_density_flow(time, lanes_count)

                self.assigned_cluster_id = cluster_algo.process(flow_change, density_change)
                # print(self.get_reward(lanes_count))

                # if self.assigned_cluster_id == 2:
                #     self.action, self.green_time = self.analytical_act(eng, time)
                # else:
=======
                self.assigned_cluster_id = cluster_algo.process(np.asarray(self.get_reward(lanes_count)), cluster_models)
                # print(self.get_reward(lanes_count))
                
>>>>>>> f7b34e639ea6bfe8e3c280fe628e8c5abed100a3
                self.action = self.act(cluster_models.model_dict[self.assigned_cluster_id][0], self.state, time, lanes_count, eps=eps)
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
<<<<<<< HEAD



    def analytical_act(self, eng, time):
        """
        selects the next action - phase for the agent to select along with the time it should stay on for
        :param eng: the cityflow simulation engine
        :param time: the time in the simulation, at this moment only integer values are supported
        :returns: the phase and the green time
        """

        self.update_clear_green_time(time)
        
        self.stabilise(time)
        if not self.action_queue.empty():
            phase, green_time = self.action_queue.get()
            return phase, int(np.ceil(green_time))

        if all([x.green_time == 0 for x in self.movements.values()]):
                return self.phase, 5
        
        self.update_priority_idx(time)
        phases_priority = {}
        
        for phase in self.phases.values():
            movements = [x for x in phase.movements if x not in self.clearing_phase.movements]

            phase_prioirty = 0
            for moveID in movements:
                phase_prioirty += self.movements[moveID].priority

            phases_priority.update({phase.ID : phase_prioirty})
        
        action = self.phases[max(phases_priority.items(), key=operator.itemgetter(1))[0]]
        if not action.movements:
            green_time = self.clearing_time
        else:
            green_time = int(np.max([self.movements[x].green_time for x in action.movements]))
        return action, green_time

    def stabilise(self, time):
        """
        Implements the stabilisation mechanism of the algorithm, updates the action queue with phases that need to be prioritiesd
        :param time: the time in the simulation, at this moment only integer values are supported
        """
        def add_phase_to_queue(priority_list):
            """
            helper function called recursievely to add phases which need stabilising to the queue
            """
            phases_score = {}
            phases_time = {}
            for elem in priority_list:
                for phaseID in elem[0].phases:
                    if phaseID in phases_score.keys():
                        phases_score.update({phaseID : phases_score[phaseID] + 1})
                    else:
                        phases_score.update({phaseID : 1})

                    if phaseID in phases_time.keys():
                        phases_time.update({phaseID : max(phases_time[phaseID], elem[1])})
                    else:
                        phases_time.update({phaseID : elem[1]})

            if [x for x in phases_score.keys() if phases_score[x] != 0]:
                idx = max(phases_score.items(), key=operator.itemgetter(1))[0]
                self.action_queue.put((self.phases[idx], phases_time[idx]))
                return [x for x in priority_list if idx not in x[0].phases]
            else:
                return []

        T = 180
        T_max = 240
        sum_Q = np.sum([x.arr_rate for x in self.movements.values()])
        
        priority_list = []

        for movement in [x for x in self.movements.values() if x.ID not in self.phase.movements]:              
            Q = movement.arr_rate
            if movement.last_on_time == -1:
                waiting_time = 0
            else:
                waiting_time = time - movement.last_on_time
                
            z = movement.green_time + movement.clearing_time + waiting_time
            n_crit = Q * T * ((T_max - z) / (T_max - T))

            waiting = movement.green_time * movement.max_saturation
           
            if waiting > n_crit:
                T_res = T * (1 - sum_Q / movement.max_saturation) - self.clearing_time * len(self.movements)
                green_max = (Q / movement.max_saturation) * T + (1 / len(self.movements)) * T_res
                priority_list.append((movement, green_max))
            
        priority_list = add_phase_to_queue(priority_list)
        # while priority_list:
        #     priority_list = add_phase_to_queue(priority_list)
=======
>>>>>>> f7b34e639ea6bfe8e3c280fe628e8c5abed100a3
