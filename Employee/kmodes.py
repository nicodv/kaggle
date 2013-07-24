#!/usr/bin/env python

'''
Implementation of the k-modes clustering algorithm.
'''
__author__  = 'Nico de Vos'
__email__   = 'njdevos@gmail.com'
__license__ = 'MIT'
__version__ = '0.2'

import random
import numpy as np
from collections import defaultdict


def kmodes(X, k, centUpd='wsample', maxIters=100, verbose=1):
    '''k-modes clustering algorithm for categorical data.
    See Huang [1997, 1998] or Chaturvedi et al. [2001].
    
    input:      X           = data points [no. attributes * no. points]
                k           = number of clusters
                centUpd     = centroid updating rule
                maxiters    = maximum iterations
    returns:    Xclust  = cluster numbers for elements of X [no. points]
                cent    = centroids [k * no. attributes]
                cost    = clustering cost
    
    NOTE: Weighted sampling in the centroid update is more accurate
    (i.e. generalizes better), but slower to converge. Taking the
    highest frequency element (which is what Huang [1998] does) converges
    faster, but is often less accurate.
    
    '''
    
    assert centUpd in ('hifreq', 'wsample')
    
    # convert to numpy array, if needed
    X = np.asanyarray(X)
    N, dim = X.shape
    assert k < N, "More clusters than data points?"
    cent = np.empty((k, dim))
    
    # ----------------------
    # INIT [see Huang, 1998]
    # ----------------------
    print("Init: sample initial centroids")
    # determine frequencies of attributes, necessary for smart init
    for idim in range(dim):
        freq = defaultdict(int)
        for at in X[:,idim]:
            freq[at] += 1
        # sample centroids using the probabilities of attributes
        # (I assume that's what's meant in the Huang [1998] paper; it works, at least)
        # note: sampling done using population in static list with as many choices as the frequency counts
        # this works well since (1) we re-use the list k times here, and (2) the counts are small
        # integers so memory consumption is low
        choices = [c for c, w in freq.items() for _ in range(w)]
        for ik in range(k):
            cent[ik, idim] = random.choice(choices)
    
    print("Init: finalizing centroids")
    # the previously chosen centroids could result in empty clusters,
    # so set centroid to closest point in X
    # TODO: when point is equally close to 2 to k clusters, randomly sample between them
    # instead of just choosing the first?
    for ik in range(k):
        dissim = get_dissim(X, cent[ik])
        ndx = dissim.argsort()
        # we want the centroid to be unique
        while np.all(X[ndx[0]] == cent, axis=1).any():
            ndx = np.delete(ndx, 0)
        cent[ik] = X[ndx[0]]
    
    # initial assigns to clusters and count attributes per cluster
    print("Init: initializing clusters")
    Xclust = np.zeros(N, dtype='int64')
    clustSize = np.zeros(k, dtype='int64')
    # clustFreq is a list of lists with dictionaries that contain the
    # frequencies of values per cluster and attribute
    clustFreq = [[defaultdict(int) for i in range(dim)] for j in range(k)]
    for ii, xx in enumerate(X):
        dissim = get_dissim(cent, xx)
        cluster = dissim.argsort()[0]
        clustSize[cluster] += 1
        Xclust[ii] = cluster
        for jj, at in enumerate(xx):
            clustFreq[cluster][jj][at] += 1
    
    # ----------------------
    # ITERATION
    # ----------------------
    print("Starting iterations...")
    iters = 0
    converged = False
    while iters <= maxIters and not converged:
        iters += 1
        moves = 0
        for ii, xx in enumerate(X):
            dissim = get_dissim(cent, xx)
            cluster = dissim.argsort()[0]
            # if necessary: move point, and update old/new cluster frequencies and centroids
            if Xclust[ii] != cluster or iters == 1:
                moves += 1
                oldcluster = Xclust[ii]
                clustSize[cluster] += 1
                clustSize[oldcluster] -= 1
                assert min(clustSize) > 0 and max(clustSize) < N-k+1
                for jj, at in enumerate(xx):
                    # update frequencies of attributes in clusters
                    clustFreq[cluster][jj][at] += 1
                    clustFreq[oldcluster][jj][at] -= 1
                    assert clustFreq[oldcluster][jj][at] >= 0
                    # update the new and old centroids by choosing (from) the most likely attribute(s)
                    for cc in (cluster, oldcluster):
                        if centUpd == 'hifreq':
                            cent[cc, jj] = key_for_max_value(clustFreq[cc][jj])
                        elif centUpd == 'wsample':
                            # like in the initialization, use the population-in-list method
                            choices = [c for c, w in clustFreq[cc][jj].items() for _ in range(w)]
                            cent[cc, jj] = random.choice(choices)
                if verbose == 2 and iters > 1:
                    print("Move from cluster {0} to {1}".format(oldcluster, cluster))
                Xclust[ii] = cluster
        
        # all points seen in this iteration
        converged = (moves == 0)
        if verbose:
            print("Iteration: {0}/{1}, moves: {2}".format(iters, maxIters, moves))
            if verbose == 2:
                print("Cluster sizes: {0}".format(clustSize))
    
    cost = clustering_cost(X, cent, Xclust)
    
    return Xclust, cent, cost

def opt_kmodes(X, k, preRuns=10, goodPctl=20, **kwargs):
    '''Shell around k-modes algorithm that tries to ensure a good clustering result
    by choosing one that has a relatively low clustering cost compared to the
    costs of a number of pre-runs. (Huang [1998] states that clustering cost can be
    used to judge the clustering quality.)

    '''
    preCosts = []
    print("Starting preruns...")
    for ii in range(preRuns):
        Xclust, cent, cost = kmodes(X, k, verbose=1, **kwargs)
        preCosts.append(cost)
        print("Cost = {0}".format(cost))
    
    while True:
        Xclust, cent, cost = kmodes(X, k, verbose=2, **kwargs)
        if cost <= np.percentile(preCosts, goodPctl):
            print("Found a good clustering.")
            print("Cost = {0}".format(cost))
            break
    
    return Xclust, cent, cost

def get_dissim(A, b):
    # TODO: other dissimilarity measures?
    # simple matching dissimilarity
    return (A != b).sum(axis=1)

def key_for_max_value(d):
    '''Very fast method (supposedly) to get key for maximum value in dict.

    '''
    v = list(d.values())
    k = list(d.keys())
    return k[v.index(max(v))]

def clustering_cost(X, clust, Xclust, **kwargs):
    '''Clustering cost, defined as the sum distance of all points
    to their respective clusters.

    '''
    cost = 0
    for ii, cc in enumerate(clust):
        cost += get_dissim(X[Xclust==ii], cc).sum()
    return cost

if __name__ == "__main__":
    # reproduce results in Huang [1998]
    
    # load small soybean disease data set
    X = np.genfromtxt('/home/nico/Code/kaggle/Employee/soybean.csv', dtype='int64', delimiter=',')[:,:-1]
    y = np.genfromtxt('/home/nico/Code/kaggle/Employee/soybean.csv', dtype='unicode', delimiter=',', usecols=35)
    
    # drop columns with single value
    X = X[:,np.std(X, axis=0) > 0.]
    
    Xclust, cent, cost = opt_kmodes(X, 4, preRuns=10, goodPctl=20, centUpd='hifreq', maxIters=40)
    
    classtable = np.zeros((4,4), dtype='int64')
    for ii,_ in enumerate(y):
        classtable[int(y[ii][-1])-1,Xclust[ii]] += 1
    
    print("    | Clust 1 | Clust 2 | Clust 3 | Clust 4 |")
    print("----|---------|---------|---------|---------|")
    for ii in range(4):
        prargs = tuple([ii+1] + list(classtable[ii,:]))
        print(" D{0} |      {1:>2} |      {2:>2} |      {3:>2} |      {4:>2} |".format(*prargs))
    
