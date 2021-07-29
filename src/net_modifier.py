import cityflow

import argparse
import os
import random
import json
import queue

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dir", default='../scenarios/4x4mount/',  type=str, help="the relative directory of the sim files")
    parser.add_argument("--sim_config", default='../scenarios/4x4mount/1.config',  type=str, help="the relative path to the simulation config file")
    parser.add_argument("--roadnet", default='../scenarios/4x4mount/roadnet.json',  type=str, help="the relative path to the simulation flow file")
    parser.add_argument("--dist_roads", default=1,  type=int, help="number of roads to be disrupted")

    return parser.parse_args()



args = parse_args()

with open(args.roadnet, "r") as roadnet_file:
    data = json.load(roadnet_file)
    for road in data['roads']:
        print(road['points'])
