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


def kmodes(X, k, init='Cao', centUpd='mode', maxIters=100, verbose=1):
    '''k-modes clustering algorithm for categorical data.
    See Huang [1997, 1998] or Chaturvedi et al. [2001].
    
    input:      X           = data points [no. attributes * no. points]
                k           = number of clusters
                init        = initialization method ('Huang' for the one described in
                              Huang [1998], 'Cao' for the more advanced one in
                              Cao et al. [2009])
                centUpd     = centroid updating method ('mode' for most common
                              attribute or 'wsample' for weighted sampling)
                maxIters    = maximum no. of iterations
    returns:    Xclust  = cluster numbers for elements of X [no. points]
                cent    = centroids [k * no. attributes]
                cost    = clustering cost
    
    '''
    
    # convert to numpy array, if needed
    X = np.asanyarray(X)
    N, at = X.shape
    assert k > 1, "Choose at least 2 clusters."
    assert k < N, "More clusters than data points?"
    
    # ----------------------
    # INIT [see Huang, 1998]
    # ----------------------
    print("Init: initializing centroids")
    cent = init_centroids(X, k, init)
    
    print("Init: initializing clusters")
    Xclust = np.zeros(N, dtype='int32')
    # clustFreq is a list of lists with dictionaries that contain the
    # frequencies of values per cluster and attribute
    clustFreq = [[defaultdict(int) for _ in range(at)] for _ in range(k)]
    for ix, curx in enumerate(X):
        # initial assigns to clusters
        dissim = get_dissim(cent, curx)
        cluster = dissim.argsort()[0]
        Xclust[ix] = cluster
        # count attribute values per cluster
        for iat, val in enumerate(curx):
            clustFreq[cluster][iat][val] += 1
    # perform an initial centroid update
    for ik in range(k):
        for iat in range(at):
            cent[ik][iat] = update_centroid(clustFreq[ik][iat], centUpd)
    
    # ----------------------
    # ITERATION
    # ----------------------
    print("Starting iterations...")
    itr = 0
    converged = False
    while itr <= maxIters and not converged:
        itr += 1
        moves = 0
        for ix, curx in enumerate(X):
            dissim = get_dissim(cent, curx)
            cluster = dissim.argsort()[0]
            # if necessary: move point, and update old/new cluster frequencies and centroids
            if Xclust[ix] != cluster:
                moves += 1
                oldcluster = Xclust[ix]
                for iat, val in enumerate(curx):
                    # update frequencies of attributes in clusters
                    clustFreq[cluster][iat][val] += 1
                    clustFreq[oldcluster][iat][val] -= 1
                    assert clustFreq[oldcluster][iat][val] >= 0
                    # update the new and old centroids by choosing (from) the most likely attribute(s)
                    for curc in (cluster, oldcluster):
                        cent[curc, iat] = update_centroid(clustFreq[curc][iat], centUpd)
                if verbose == 2:
                    print("Move from cluster {0} to {1}".format(oldcluster, cluster))
                Xclust[ix] = cluster
        
        # all points seen in this iteration
        converged = (moves == 0)
        if verbose:
            print("Iteration: {0}/{1}, moves: {2}".format(itr, maxIters, moves))
    
    cost = clustering_cost(X, cent, Xclust)
    
    return Xclust, cent, cost

def init_centroids(X, k, init='Cao'):
    assert init in ('Huang', 'Cao')
    N, at = X.shape
    cent = np.empty((k, at))
    if init == 'Huang':
        # determine frequencies of attributes
        for iat in range(at):
            freq = defaultdict(int)
            for val in X[:,iat]:
                freq[val] += 1
            # sample centroids using the probabilities of attributes
            # (I assume that's what's meant in the Huang [1998] paper; it works, at least)
            # note: sampling done using population in static list with as many choices as the frequency counts
            # this works well since (1) we re-use the list k times here, and (2) the counts are small
            # integers so memory consumption is low
            choices = [chc for chc, wght in freq.items() for _ in range(wght)]
            for ik in range(k):
                cent[ik, iat] = random.choice(choices)
        # the previously chosen centroids could result in empty clusters,
        # so set centroid to closest point in X
        for ik in range(k):
            dissim = get_dissim(X, cent[ik])
            ndx = dissim.argsort()
            # we want the centroid to be unique
            while np.all(X[ndx[0]] == cent, axis=1).any():
                ndx = np.delete(ndx, 0)
            cent[ik] = X[ndx[0]]
    elif init == 'Cao':
        # Note: O(N * at * k**2), so watch out with k
        # determine densities points
        dens = np.zeros(N)
        for iat in range(at):
            freq = defaultdict(int)
            for val in X[:,iat]:
                freq[val] += 1
            for iN in range(N):
                dens[iN] += freq[X[iN,iat]] / float(at)
        dens /= N
        
        # choose centroids based on distance and density
        cent[0] = X[np.argmax(dens)]
        dissim = get_dissim(X, cent[0])
        cent[1] = X[np.argmax(dissim * dens)]
        # for the reamining centroids, choose max dens * dissim to the (already assigned)
        # centroid with the lowest dens * dissim
        for ic in range(2,k):
            dd = np.empty((ic, N))
            for icp in range(ic):
                dd[icp] = get_dissim(X, cent[icp]) * dens
            cent[ic] = X[np.argmax(np.min(dd, axis=0))]
    
    return cent

def update_centroid(freqs, centUpd):
    # TODO: Taking the mode (i.e. highest frequency element), which is what Huang [1998]
    # suggests, converges faster and takes less processing power. However, weighted sampling
    # in the centroid update makes intuitive sense to me and might offer better
    # generalization. Investigate.
    assert centUpd in ('mode', 'wsample')
    if centUpd == 'mode':
        return key_for_max_value(freqs)
    elif centUpd == 'wsample':
        choices = [chc for chc, wght in freqs.items() for _ in range(wght)]
        return random.choice(choices)

def get_dissim(A, b):
    # TODO: add other dissimilarity measures?
    # simple matching dissimilarity
    return (A != b).sum(axis=1)

def clustering_cost(X, clust, Xclust, **kwargs):
    '''Clustering cost, defined as the sum distance of all points
    to their respective clusters.

    '''
    cost = 0
    for ic, curc in enumerate(clust):
        cost += get_dissim(X[Xclust==ic], curc).sum()
    return cost

def key_for_max_value(d):
    '''Very fast method (supposedly) to get key for maximum value in dict.

    '''
    v = list(d.values())
    k = list(d.keys())
    return k[v.index(max(v))]

def opt_kmodes(X, k, preRuns=10, goodPctl=20, **kwargs):
    '''Shell around k-modes algorithm that tries to ensure a good clustering result
    by choosing one that has a relatively low clustering cost compared to the
    costs of a number of pre-runs. (Huang [1998] states that clustering cost can be
    used to judge the clustering quality.)

    '''
    
    if kwargs['init'] == 'Cao' and kwargs['centUpd'] == 'mode':
        print("""Hint: Cao initialization method + mode updates = deterministic.
                No opt_kmodes necessary, run kmodes method directly instead.""")
    
    preCosts = []
    print("Starting preruns...")
    for _ in range(preRuns):
        Xclust, cent, cost = kmodes(X, k, verbose=0, **kwargs)
        preCosts.append(cost)
        print("Cost = {0}".format(cost))
    
    while True:
        Xclust, cent, cost = kmodes(X, k, verbose=1, **kwargs)
        if cost <= np.percentile(preCosts, goodPctl):
            print("Found a good clustering.")
            print("Cost = {0}".format(cost))
            break
    
    return Xclust, cent, cost

if __name__ == "__main__":
    # reproduce results in Huang [1998]
    
    # load small soybean disease data set
    X = np.genfromtxt('/home/nico/Code/kaggle/Employee/soybean.csv', dtype='int64', delimiter=',')[:,:-1]
    y = np.genfromtxt('/home/nico/Code/kaggle/Employee/soybean.csv', dtype='unicode', delimiter=',', usecols=35)
    
    # drop columns with single value
    X = X[:,np.std(X, axis=0) > 0.]
    
    #Xclust, cent, cost = kmodes(X, 4, init='Huang', centUpd='mode', maxIters=100)
    Xclust, cent, cost = opt_kmodes(X, 4, preRuns=10, goodPctl=20, init='Huang', centUpd='mode', maxIters=100)
    
    classtable = np.zeros((4,4), dtype='int64')
    for ii,_ in enumerate(y):
        classtable[int(y[ii][-1])-1,Xclust[ii]] += 1
    
    print("    | Clust 1 | Clust 2 | Clust 3 | Clust 4 |")
    print("----|---------|---------|---------|---------|")
    for ii in range(4):
        prargs = tuple([ii+1] + list(classtable[ii,:]))
        print(" D{0} |      {1:>2} |      {2:>2} |      {3:>2} |      {4:>2} |".format(*prargs))
    
