from SOStream.utils import dist, weighted_mean
from SOStream.micro_cluster import MicroCluster
import numpy as np

def merge_clusters(win_micro_cluster, overlaping_micro_clusters, merge_threshold, cluster_models):
    merged_cluster = None
    deleted_clusters = list()
    for micro_cluster in overlaping_micro_clusters:
        if dist(micro_cluster.centroid, win_micro_cluster.centroid) < merge_threshold:
            if len(deleted_clusters) == 0:
                deleted_clusters.append(win_micro_cluster)
                merged_cluster = MicroCluster(win_micro_cluster.centroid, data=win_micro_cluster.data,
                                              number_points=win_micro_cluster.number_points,
                                              radius=win_micro_cluster.radius, ID=win_micro_cluster.ID)

            merged_cluster = merge(micro_cluster, merged_cluster, cluster_models)
            deleted_clusters.append(micro_cluster)
    return merged_cluster, deleted_clusters


def merge(cluster_a, cluster_b, cluster_models):
    new_cluster_centroid = weighted_mean(cluster_a.centroid, cluster_b.centroid, cluster_a.number_points, cluster_b.number_points)
    new_cluster_radius = dist(cluster_a.centroid, cluster_b.centroid) + max(cluster_a.radius, cluster_b.radius)

    new_cluster_data = cluster_a.data + cluster_b.data

    if cluster_b.number_points > cluster_a.number_points:
        new_cluster_id = cluster_b.ID
    else:
        new_cluster_id = cluster_a.ID

    new_cluster = MicroCluster(centroid=new_cluster_centroid, data=new_cluster_data,
                               number_points=cluster_a.number_points + cluster_b.number_points,
                               radius=new_cluster_radius, ID=new_cluster_id)

    cluster_models.merge_models(cluster_a.ID, cluster_b.ID)
  
    return new_cluster
