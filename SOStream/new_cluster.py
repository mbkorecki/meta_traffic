from SOStream.micro_cluster import MicroCluster

def newCluster(vt):
    return MicroCluster(centroid=vt, data=[vt])

