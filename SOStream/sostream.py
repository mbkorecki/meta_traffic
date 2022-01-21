from SOStream.utils import min_dist, dist
from SOStream.find_neighbors import find_neighbors
from SOStream.update_cluster import updateCluster
from SOStream.find_overlap import find_overlap
from SOStream.merge_clusters import merge_clusters
from SOStream.new_cluster import newCluster


class SOStream:

    def __init__(self, alpha = 0.1, min_pts = 10, merge_threshold = 27000):
        self.alpha = alpha
        self.min_pts = min_pts
        self.M = [[]]
        self.merge_threshold = merge_threshold

    def process(self, vt):
        winner_micro_cluster = min_dist(vt, self.M[-1])        
        new_M = self.M[-1].copy()
        assigned_cluster = None
        
        if len(new_M) >= self.min_pts:
            winner_neighborhood = find_neighbors(winner_micro_cluster, self.min_pts, new_M)
            if dist(vt, winner_micro_cluster.centroid) < winner_micro_cluster.radius:
                updateCluster(winner_micro_cluster, vt, self.alpha, winner_neighborhood)
                assigned_cluster = winner_micro_cluster
            else:
                new_M.append(newCluster(vt))
                assigned_cluster = new_M[-1]

            overlap = find_overlap(winner_micro_cluster, winner_neighborhood)
            if len(overlap) > 0:
                merged_cluster, deleted_clusters = merge_clusters(winner_micro_cluster, overlap, self.merge_threshold)
                for deleted_cluster in deleted_clusters:
                    new_M.remove(deleted_cluster)
                if merged_cluster is not None:
                    new_M.append(merged_cluster)
                    assigned_cluster = merged_cluster
        else:
            new_M.append(newCluster(vt))
            assigned_cluster = new_M[-1]

        self.M.append(new_M)
        # print('clustering ', len(assigned_cluster.memory), len(assigned_cluster.data))
        return assigned_cluster

    pass
