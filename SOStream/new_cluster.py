from SOStream.micro_cluster import MicroCluster

def newCluster(vt, ID):
    return MicroCluster(centroid=vt, data=[vt], ID=ID)

