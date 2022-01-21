from SOStream.sostream import SOStream
from dqn import DQN, ReplayMemory
from collections import namedtuple, deque 
import random
import numpy as np

np.random.seed(23)

clustering = SOStream(alpha = 0, min_pts = 3, merge_threshold = 50)
data1 = np.random.uniform(1000, 1001, size=(100,1))
data2 = np.random.uniform(1, 2, (100,1))

data = np.concatenate((data1, data2), axis=0)

for d in data:
    clustering.process(d)

s1 = np.array([c.centroid for c in clustering.M[-1]])
    
for c in clustering.M[-1]:
    print(c.centroid, c.radius, c.number_points, len(c.data))



# self.clustering = SOStream(alpha=0.1, min_pts=64, merge_threshold=100)
# #clustering based on state or reward
# assgined_cluster = self.clustering.process(reward)
# assigned_cluster.memory.add()

    
