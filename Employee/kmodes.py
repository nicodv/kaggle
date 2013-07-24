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

def kmodes(X, k, dissimMeas='simple', centUpd='wsample', maxIters=100, verbose=1):
    '''k-modes clustering algorithm for categorical data.
    See Huang [1997, 1998] or Chaturvedi et al. [2001].
    
    input:      X           = data points [no. attributes * no. points]
                k           = number of clusters
                dissimMeas  = dissimilarity measure
                centUpd     = centroid updating rule
                maxiters    = maximum iterations
    returns:    Xclust  = cluster numbers for elements of X [no. points]
                cent    = centroids [k * no. attributes]
                cost    = clustering cost
    
    NOTE: Weighted sampling in the centroid update is more accurate
    (i.e. generalizes better), but slower to converge. Taking the
    highest frequency element (which is what Huang [1998] does) converges
    faster, but is often less accurate.
    NOTE: Dissimilarity measure 'weighted' takes into account relative
    frequencies of attributes (see He et al. [2005] or San et al. [2004]).
    
    '''
    
    assert dissimMeas in ('simple','weighted')
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
        choices = [c for c, w in freq for _ in range(w)]
        for ik in range(k):
            cent[ik, idim] = random.choice(choices)
    
    print("Init: finalizing centroids")
    # the previously chosen centroids could result in empty clusters,
    # so set centroid to closest point in X
    # TODO: when point is equally close to 2 to k clusters, randomly sample between them
    # instead of just choosing the first?
    for ik in range(k):
        dissim = get_dissim(X, cent[ik], 'simple')
        ndx = dissim.argsort()
        # we want the centroid to be unique
        while X[ndx[0]] in cent:
            ndx = ndx.delete(0)
        cent[ik] = X[ndx[0]]
    
    # initial assigns to clusters and count attributes per cluster
    print("Init: initializing clusters")
    Xclust = np.zeros(N, dtype='int64')
    clustSize = [0]*k
    clustFreq = [defaultdict(int)]*k
    for ii, xx in enumerate(X):
        dissim = get_dissim(cent, xx, 'simple')
        cluster = dissim.argsort()[0]
        clustSize[cluster] += 1
        Xclust[ii] = cluster
        for at in xx:
            clustFreq[cluster][at] += 1
    
    # ----------------------
    # ITERATION
    # ----------------------
    print("Starting iterations...")
    iters = 0
    converged = False
    while iters <= maxIters and not converged:
        iters += 1
        moves = 0
        for ii, xx in X:
            dissim = get_dissim(cent, xx, dissimMeas, clustSize, clustFreq)
            cluster = dissim.argsort()[0]
            # if necessary: move point, and update old/new cluster frequencies and centroids
            if Xclust[ii] != cluster:
                moves += 1
                oldcluster = Xclust[ii]
                clustSize[cluster] += 1
                clustSize[oldcluster] -= 1
                assert min(clustSize) > 0 and max(clustSize) < N-k+1
                for at in xx:
                    # update frequencies of attributes in clusters
                    clustFreq[cluster][at] += 1
                    clustFreq[oldcluster][at] -= 1
                    assert clustFreq[oldcluster][at] >= 0
                    # update the new and old centroids by choosing (from) the most likely attribute(s)
                    for cc in (cluster, oldcluster):
                        if centUpd == 'hifreq':
                            cent[cc, idim] = key_for_max_value(clustFreq[cc])
                        elif centUpd == 'wsample':
                            # like in the initialization, use the population-in-list method
                            choices = [c for c, w in clustFreq[cc] for _ in range(w)]
                            cent[cc, idim] = random.choice(choices)
                Xclust[ii] = cluster
                if verbose == 2:
                    print("Move from cluster {0} to {1}".format(Xclust[ii], cluster))
        
        # all points seen in this iteration
        converged = (moves == 0)
        if verbose:
            print("Iteration: {0}/{1}, moves: {2}".format(iters, maxIters, moves))
            if verbose == 2:
                print("Cluster sizes: {0}".format(clustSize))
    
    cost = clustering_cost(X, cent, clustSize, clustFreq)
    
    return Xclust, cent, cost

def opt_kmodes(X, k, preruns=10, goodpctl=20, **kwargs):
    '''Shell around k-modes algorithm that tries to ensure a good clustering result
    by choosing one that has a relatively low clustering cost compared to the
    costs of a number of pre-runs. (Huang [1998] states that clustering cost can be
    used to judge the clustering quality.)

    '''
    preCosts = []
    print("Starting preruns...")
    for ii in range(preRuns):
        Xclust, cent, cost = kmodes(X, k, **kwargs, verbose=0)
        preCosts.append(cost)
        print("Cost = {0}".format(cost))
    
    while True:
        Xclust, cent, cost = kmodes(X, k, **kwargs, verbose=1)
        if cost <= np.percentile(preCosts, goodPctl):
            print("Found a good clustering.")
            print("Cost = {0}".format(cost))
            break
    
    return Xclust, cent, cost

def get_dissim(A, b, dissimMeas, sizes=None, freqs=None):
    '''Return dissimilarity measure between a number of points A and point b.
    
    '''
    assert dissimMeas in ('simple', 'weighted')
    assert (dissimMean == 'weighted') == (sizes and freqs)
    
    if dissimMeas == 'simple':
        return (A != b).sum(axis=1)
    elif dissimMeas == 'weighted':
        # unequal:  dissim = 1 (just like simple)
        # equal:    dissim = 1 - no. of same attributes of points in cluster / number of points in cluster
        dissim = []
        for ii, cur in enumerate(A):
            # values to subtract for equal attributes
            substr = [freqs[cur][x] / sizes[ii] for x in cur[cur == b]]
            dissim.append(len(cur) - sum(substr))
        return np.array(dissim)

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
        cost += get_dissim(X[Xclust==ii], cc, **kwargs).sum()
    return cost

if __name__ == "__main__":
    # reproduce results in Huang [1998]
    
    # load small soybean disease data set
    X = np.genfromtxt('/home/nico/Code/kaggle/Employee/soybean.csv', dtype='int64', delimiter=',')[:,:-1]
    y = np.genfromtxt('/home/nico/Code/kaggle/Employee/soybean.csv', dtype='unicode', delimiter=',', usecols=35)
    
    # drop columns with single value
    X = X[:,np.std(X, axis=0) > 0.]
    
    Xclust, cent, cost = opt_kmodes(X, 4, preRuns=10, goodPctl=20, dissimMeas='simple',
                                    centUpd='hifreq', maxIters=40)
    
    classtable = np.zeros((4,4), dtype='int64')
    for ii,_ in enumerate(y):
        classtable[int(y[ii][-1])-1,Xclust[ii]] += 1
    
    print("    | Clust 1 | Clust 2 | Clust 3 | Clust 4 |")
    print("----|---------|---------|---------|---------|")
    for ii in range(4):
        prargs = tuple([ii+1] + list(classtable[ii,:]))
        print(" D{0} |      {1:>2} |      {2:>2} |      {3:>2} |      {4:>2} |".format(*prargs))
    
