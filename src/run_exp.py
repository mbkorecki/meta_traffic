import os
import argparse
import json
import numpy as np
import random
from datetime import datetime

vehicle = {
    "length": 5.0,
    "width": 2.0,
    "maxPosAcc": 2.0,
    "maxNegAcc": 4.5,
    "usualPosAcc": 2.0,
    "usualNegAcc": 4.5,
    "minGap": 2.5,
    "maxSpeed": 11.11,
    "headwayTime": 1.5
}

def generate_route(turning_ratios, sim_time, road_link_dict, road_len_dict, start_road_list):
    route = []
    options = ['turn_left', 'turn_right', 'go_straight']
    total_time = 0

    current_road = random.choice(start_road_list)
    time_to_cross = (road_len_dict[current_road] / 20)
    total_time += time_to_cross
    route.append(current_road)
    
    while total_time < sim_time:
        if current_road not in road_link_dict.keys():
            break
        direction = random.choices(list(options), weights=turning_ratios)[0]
        current_road = road_link_dict[current_road][direction]
        time_to_cross = (road_len_dict[current_road] / 20)
        total_time += time_to_cross
        route.append(current_road)

    return route

def generate_flow_file(path, roadnet, num_vehs, turning_ratios, sim_time):
    with open(roadnet, "r") as roadnet_file:
        roadnet_data = json.load(roadnet_file)

    flow_data = []
    
    road_len_dict = {}
    road_link_dict = {}
    start_road_list = []
    
    for road in roadnet_data['roads']:
        point1 = road['points'][0]
        point2 = road['points'][1]

        diff_x = point1['x'] - point2['x']
        diff_y = point1['y'] - point2['y']

        length = np.sqrt(diff_x**2 + diff_y**2)
        road_len_dict[road['id']] = length

    for intersection in roadnet_data['intersections']:
        if intersection['virtual']:
            for road in intersection['roads']:
                my_road = [x for x in roadnet_data['roads'] if x['id'] == road]
                if my_road != [] and my_road[0]['startIntersection'] == intersection["id"]:
                    start_road_list.append(road)
        for link in intersection['roadLinks']:
            if link["startRoad"] not in road_link_dict.keys():
                road_link_dict[link["startRoad"]] = {link["type"] : link["endRoad"]}
            else:
                road_link_dict[link["startRoad"]].update({link["type"] : link["endRoad"]})

         

    random.seed(datetime.now())
    for veh in range(num_vehs):
        veh_data = {'vehicle' : vehicle}
        veh_data['interval'] = 1
        veh_data['startTime'] = random.randrange(0, 200)
        veh_data['endTime'] = veh_data['startTime']
        route = generate_route(turning_ratios, sim_time, road_link_dict, road_len_dict, start_road_list)
        veh_data['route'] = route
        flow_data.append(veh_data)

    save_path = path + "flow.json"
    i = 0
    while os.path.exists(save_path):
        save_path = path + "flow(" + str(i) + ").json"
        i += 1
        
    with open(save_path, 'w') as flow:
        json.dump(flow_data, flow)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--roadnet", default='../scenarios/1x1sphere/roadnet.json',  type=str, help="the relative path to the simulation roadnet file")
    parser.add_argument("--num_vehs", default=30,  type=int, help="the number of vehicles introduced into the system")
    parser.add_argument("--agent_type", default='random',  type=str, help="the number of vehicles introduced into the system")

    return parser.parse_args()


args = parse_args()
print(args)


#create a folder for the experiment

path = '../exp'
i = 0
while os.path.exists(path):
    path = "../exp(" + str(i) + ")"
    i += 1

os.mkdir(path)
os.mkdir(path + '/scenarios')
os.mkdir(path + '/results')


config = {"interval": 1, "seed": 0, "dir": path + '/scenarios/', "roadnetFile": "../roadnet.json", "flowFile": "flow.json", "rlTrafficLight": True, "saveReplay": True, "roadnetLogFile": "roadnetLogFile.json", "replayLogFile": "replayLogFile.txt", "laneChange": False}


with open(args.roadnet, 'r') as roadnet_file:
    roadnet_data = json.load(roadnet_file)
    with open(path + '/scenarios/roadnet.json', 'w') as roadnet:
        json.dump(roadnet_data, roadnet)
    

if args.agent_type == 'hybrid' or args.agent_type == 'cluster':
    # TRAINING
    for i in range(100):
        os.mkdir(path + '/scenarios/' + str(i) + '/')
        generate_flow_file(path + '/scenarios/' + str(i) + '/', args.roadnet, args.num_vehs, [1, 1, 1], 1800)
        config['dir'] = path + '/scenarios/' + str(i) + '/'

        with open(path + '/scenarios/' + str(i) + '/' + str(i) + '.config', 'w') as config_file:
            json.dump(config, config_file)
    
    os.system("python traffic_sim.py --meta True --sim_config \"" +  path + '/scenarios/' + str(0) + "/" + str(0) + ".config\"" + " --num_sim_steps 1800 --num_episodes 100 --lr 0.0005 --agents_type " + args.agent_type + " --path \"" + path + '/results\"')


    #TESTING
    for i in range(1):
        print("test " + str(i))
        os.system("python traffic_sim.py --sim_config \"" +  path + '/scenarios/' + str(i) + "/" + str(i) + ".config\"" + " --num_sim_steps 1800 --num_episodes 1 --lr 0.0005 --agents_type " + args.agent_type + " --path \"" + path + '/results\"' + " --load \"" + path + '/results/scenarios_config0_' + args.agent_type  +'/time_target_net.pt\"'
              + " --mode \"test\" --eps_start 0 --eps_end 0")

    
else:
    for i in range(1):
        os.mkdir(path + '/scenarios/' + str(i) + '/')
        generate_flow_file(path + '/scenarios/' + str(i) + '/', args.roadnet, args.num_vehs, [1, 1, 1], 1800)
        config['dir'] = path + '/scenarios/' + str(i) + '/'

        with open(path + '/scenarios/' + str(i) + '/' + str(i) + '.config', 'w') as config_file:
            json.dump(config, config_file)

        print("test " + str(i))
        os.system("python traffic_sim.py --sim_config \"" +  path + '/scenarios/' + str(i) + "/" + str(i) + ".config\"" + " --num_sim_steps 1800 --num_episodes 1 --lr 0.0005 --agents_type " + args.agent_type + " --path \"" + path + '/results\"')  
