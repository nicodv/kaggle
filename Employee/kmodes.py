#!/usr/bin/env python
import sys, math, random
import FileParser

tt_vals = None

def find_distance(cur_sample, centroid):
    dist_vals = []
    for i in range(34):
        dist_vals_r = []
        for j in range(7128):
            d = centroid - cur_sample[i][j]
            dist_vals_r.append(abs(d))
        dist_vals.append(dist_vals_r)
    return dist_vals

def find_centroid(cur_sample):
    cnt = get_most_commons(cur_sample)
    centroid = cur_sample[cnt]
    return centroid
    
def get_smallest(data_list):
    smallest = data_list[0]
    for val in data_list:
        if val < smallest:
            smallest = val
    return data_list.index(smallest)
    
def build_cluster(k_bc, dist_vals_all_bc, tt_vals_bc):
    clust_bc = []
    for i in range(k_bc):
        clust_bc.append([])
    for i in range(34):
        for j in range(7128):
            argm_list = []
            for k in range(k_bc):
                argm_list.append(dist_vals_all_bc[k][i][j])
            
            smalest_dist_val = get_smallest(argm_list)
            clust_bc[smalest_dist_val].append([i, j])
        
    return clust_bc

def find_distance_for_all(t_random_fn_dist, tt_vals_fn_dist):
    dist_vals_all_fn = []
    for cur_k in t_random_fn_dist:
        cur_dist_vals = find_distance(tt_vals_fn_dist, cur_k)
        dist_vals_all_fn.append(cur_dist_vals)
    return dist_vals_all_fn

def reconstruct_clusters(clusters_fn, tt_vals_fn):
    centriods_list = []
    k = 0
    for clust in clusters_fn:
        if len(clust) > 0:
            cent_coords = find_centroid(clust)
            x = cent_coords[0]
            y = cent_coords[1]
            centriods_list.append(tt_vals[x][y])
            k = k + 1
    #print centriods_list
    dist_vals_all = find_distance_for_all(centriods_list, tt_vals_fn)
    clusts = build_cluster(k, dist_vals_all, tt_vals_fn)
    return clusts, centriods_list

def get_most_commons(mylist):
    mylist_vals = []
    for ml in mylist:
        mylist_vals.append(tt_vals[ml[0]][ml[1]])
    import sys
    sys.path.append("D:\python27\lib")
    import collections
    x = collections.Counter(mylist_vals)
    k = x.most_common()
    m = (0,-99)
    for j in k:
        if(j[1]>m[1]):
            m=j
    return mylist_vals.index(m[0])

def main(k = 3):
    global tt_vals
    print "Started Building clusters... Requested No. of clusters are: ", k
    tt_obj = FileParser.FileParser("luekemia.csv")
    tt_vals = tt_obj.parse_file()
    
    dist_vals_all = []
    
    clusters = []
    for i in range(k):
        for j in range(k):
            clusters.append([])
    
    random_i_list = random.sample(xrange(1, 34), k)
    random_j_list = random.sample(xrange(7128), k)
    
    #print random_i_list, random_j_list
    
    t_random = []
    for p in range(k):
        t_random.append(tt_vals[random_i_list[p]][random_j_list[p]])
    
    print "Centroid: ", t_random
    
    dist_vals_all = find_distance_for_all(t_random, tt_vals)
    clusts = build_cluster(k, dist_vals_all, tt_vals)
    
    centriods = t_random
    
    while True:
        prev_centriods = centriods
        clusts, centriods = reconstruct_clusters(clusts, tt_vals)
        print "Centroid: ", centriods
        flg = 0
        for cur_centroid in centriods:
            if not cur_centroid in prev_centriods:
                flg = 1
        if flg == 0:
            break
        
    print "\n\nFinal Centriods", centriods
    for i in xrange(k):
        print "Number of items in Cluster ", i + 1, ": ", len(clusts[i])
    
    for i in xrange(k):
        print "Items in Cluster ", i + 1, ": ", clusts[i][0:4]," ... ", clusts[i][-4:]
    
    AML_count_in_clusters = []
    ALL_count_in_clusters = []
    
    for single_cluster in clusts:
        cur_clust_AML_count = 0
        cur_clust_ALL_count = 0
        for coords in single_cluster:
            if tt_vals[coords[0]][7129] == "AML":
                cur_clust_AML_count = cur_clust_AML_count + 1
            else:
                cur_clust_ALL_count = cur_clust_ALL_count + 1
        AML_count_in_clusters.append(cur_clust_AML_count)
        ALL_count_in_clusters.append(cur_clust_ALL_count)
        
    FH = open("finalOutPut.csv", "a")
    
    for i in range(k):
        print "\nCluster #: %d" % (i + 1)
        print "\tAMLs : %d" % (AML_count_in_clusters[i])
        print "\tALLs : %d" % (ALL_count_in_clusters[i])
        outputstring = "str(k)," + str(i + 1) + "," + str(AML_count_in_clusters[i]) + "," + str(ALL_count_in_clusters[i]) + "\n"
        FH.write(outputstring)
        FH.flush()
    
    FH.close()

if __name__ == '__main__':
    main(int(sys.argv[1]))
