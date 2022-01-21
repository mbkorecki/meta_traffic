from SOStream.utils import dist, weighted_mean
from SOStream.micro_cluster import MicroCluster
import numpy as np

def merge_clusters(win_micro_cluster, overlaping_micro_clusters, merge_threshold):
    merged_cluster = None
    deleted_clusters = list()
    for micro_cluster in overlaping_micro_clusters:
        if dist(micro_cluster.centroid, win_micro_cluster.centroid) < merge_threshold:
            if len(deleted_clusters) == 0:
                deleted_clusters.append(win_micro_cluster)
                merged_cluster = MicroCluster(win_micro_cluster.centroid, data=win_micro_cluster.data,
                                              number_points=win_micro_cluster.number_points,
                                              radius=win_micro_cluster.radius, memory=win_micro_cluster.memory,
                                              local_net=win_micro_cluster.local_net, target_net=win_micro_cluster.target_net,
                                              lr=win_micro_cluster.optimizer.param_groups[0]["lr"])
            merged_cluster = merge(micro_cluster, merged_cluster)
            deleted_clusters.append(micro_cluster)
    return merged_cluster, deleted_clusters


def merge(cluster_a, cluster_b):
    new_cluster_centroid = weighted_mean(cluster_a.centroid, cluster_b.centroid, cluster_a.number_points, cluster_b.number_points)
    new_cluster_radius = dist(cluster_a.centroid, cluster_b.centroid) + max(cluster_a.radius, cluster_b.radius)

    new_cluster_data = cluster_a.data + cluster_b.data
    new_cluster_memory = cluster_a.memory + cluster_b.memory
    
    new_cluster = MicroCluster(centroid=new_cluster_centroid, data=new_cluster_data,
                               number_points=cluster_a.number_points + cluster_b.number_points,
                               radius=new_cluster_radius, memory=new_cluster_memory,
                               lr=np.mean([cluster_a.optimizer.param_groups[0]["lr"], cluster_b.optimizer.param_groups[0]["lr"]]))

    # print(cluster_a.optimizer.param_groups[0]["lr"], cluster_b.optimizer.param_groups[0]["lr"])
    param_a = cluster_a.local_net.state_dict()
    param_b = cluster_b.local_net.state_dict()
    weight_a = len(cluster_a.data) / (len(cluster_a.data) + len(cluster_b.data))
    weight_b = len(cluster_b.data) / (len(cluster_a.data) + len(cluster_b.data))
    
    for key in param_a:
        param_b[key] = (param_a[key] + param_b[key]) / 2.

    new_cluster.local_net.load_state_dict(param_b)

    param_a = cluster_a.target_net.state_dict()
    param_b = cluster_b.target_net.state_dict()
    for key in param_a:
        param_b[key] = (weight_a * param_a[key] + weight_b * param_b[key]) / 2.

    new_cluster.target_net.load_state_dict(param_b)
  
    return new_cluster
