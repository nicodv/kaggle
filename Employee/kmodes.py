#!/usr/bin/env python

'''
Implementation of the k-modes clustering algorithm by Huang [1997, 1998] / Chaturvedi et al. [2001].
'''
__author__  = 'N.J. de Vos'
__email__   = 'njdevos@gmail.com'
__license__ = 'MIT'
__version__ = '0.1'


import numpy as np

def kmodes(X, k, centUpd='wsampling', maxiters=100, verbose=1):
    '''k-modes clustering algorithm for categorical data.
    See Huang [1997, 1998] or Chaturvedi et al. [2001].
    
    input:      X = data points (no. attributes * no. points)
                k = number of clusters
                centUpd = centroid updating rule
                maxiters = maximum iterations
    returns:    Xclust = cluster numbers for elements of X
                cent = centroids
                cost = clustering cost
    
    NOTE: Weighted sampling in the centroid update is more accurate
    (i.e. generalizes better), but slower to converge. Taking the
    highest frequency element (which is what Huang does) converges
    faster, but is often less accurate.
    
    '''
    
    # convert to numpy array, if needed
    X = np.asanyarray(X)
    N, dim = X.shape
    assert k < N, "More clusters than data points?"
    cent = np.empty((k, dim))
    
    # ----------------------
    # INIT [see Huang, 1998]
    # ----------------------
    print("Init: counting elements")
    # determine total frequencies of attributes, necessary for smart init
    freqs = {}
    for idim in range(dim):
        for iN in range(N):
            if freqs.has_key((idim,X[iN,idim])):
                freqs[(idim,X[iN,idim])] += 1
            else:
                freqs[(idim,X[iN,idim])] = 1
    
    print("Init: sample centroids")
    # sample centroids using the probabilities of attributes
    # (I assume that's what's meant in the Huang [1998] paper; it seems to work, at least)
    for ik in range(k):
        for idim in range(dim):
            choices = [x[1] for x in freqs if x[0] == idim]
            weights = [freqs[(x[0],x[1])] for x in freqs if x[0] == idim]
            cent[ik, idim] = weighted_choice(zip(choices, weights))
    
    print("Init: finalizing centroids")
    # the previously chosen centroids could result in empty clusters,
    # so set centroid to closest point in X
    minInds = []
    for ik in range(k):
        minDist = np.inf
        for iN in range(N):
            # check this, because we don't want two same centroids
            if iN not in minInds:
                dist = get_distance(X[iN], cent[ik])
                if dist < minDist:
                    minDist = dist
                    minInd = iN
        minInds.append(minInd)
        cent[ik] = X[minInd]
    
    # initial assigns to clusters and count attributes per cluster
    print("Init: initializing clusters")
    Xclust = np.zeros(N, dtype='int64')
    freqClust = {}
    for iN in range(N):
        minDist = np.inf
        for ik in range(k):
            dist = get_distance(X[iN], cent[ik])
            if dist < minDist:
                minDist = dist
                cluster = ik
        Xclust[iN] = cluster
        for idim in range(dim):
            if freqClust.has_key((cluster, idim, X[iN, idim])):
                freqClust[(cluster, idim, X[iN, idim])] += 1
            else:
                freqClust[(cluster, idim, X[iN, idim])] = 1
    
    # ----------------------
    # ITERATION
    # ----------------------
    print("Starting iterations...")
    iters = 0
    converged = False
    while iters <= maxiters and not converged:
        iters += 1
        moves = 0
        for iN in range(N):
            minDist = np.inf
            for ik in range(k):
                dist = get_distance(X[iN], cent[ik])
                if dist < minDist:
                    minDist = dist
                    cluster = ik
            # if necessary, move point and update centroids
            if Xclust[iN] != cluster or iters == 1:
                moves += 1
                oldcluster = Xclust[iN]
                for idim in range(dim):
                    # update frequencies of attributes in clusters
                    if freqClust.has_key((cluster, idim, X[iN, idim])):
                        freqClust[(cluster, idim, X[iN, idim])] += 1
                    else:
                        freqClust[(cluster, idim, X[iN, idim])] = 1
                    freqClust[(oldcluster, idim, X[iN, idim])] -= 1
                    # update the centroids by choosing (from) the most likely attribute(s)
                    ccounts = [freqClust[(x[0],x[1],x[2])] for x in freqClust if x[0] == cluster and x[1] == idim]
                    cvalues = [x[2] for x in freqClust if x[0] == cluster and x[1] == idim]
                    if centUpd == 'wsampling':
                        cent[cluster, idim] = weighted_choice(zip(cvalues, ccounts))
                    elif centUpd == 'highest':
                        cent[cluster, idim] = cvalues[np.argmax(ccounts)]
                    occounts = [freqClust[(x[0],x[1],x[2])] for x in freqClust if x[0] == oldcluster and x[1] == idim]
                    ocvalues = [x[2] for x in freqClust if x[0] == oldcluster and x[1] == idim]
                    if centUpd == 'wsampling':
                        cent[oldcluster, idim] = weighted_choice(zip(ocvalues, occounts))
                    elif centUpd == 'highest':
                        cent[oldcluster, idim] = ocvalues[np.argmax(occounts)]
                Xclust[iN] = cluster
        converged = (moves == 0)
        if verbose:
            print("Iteration: {0}, moves: {1}".format(iters, moves))
    
    cost = clustering_cost(X, cent, Xclust)
    
    return Xclust, cent, cost

def opt_kmodes(X, k, preruns=10, goodpctl=20, **kwargs):
    '''Shell around k-modes algorithm that tries to ensure a good clustering result
    by choosing one that has a relatively low clustering cost compared to the
    costs of a number of pre-runs. (Huang [1998] states that clustering cost can be
    used to judge the clustering quality.)
    
    '''
    precosts = []
    print("Starting preruns...")
    for ii in range(preruns):
        Xclust, cent, cost = kmodes(X, k, kwargs['centUpd'], kwargs['maxiters'], verbose=1)
        precosts.append(cost)
        print("Cost = {0}".format(cost))
    
    while True:
        Xclust, cent, cost = kmodes(X, k, kwargs['centUpd'], kwargs['maxiters'], verbose=1)
        if cost <= np.percentile(precosts, goodpctl):
            print("Found a good clustering.")
            print("Cost = {0}".format(cost))
            break
    
    return Xclust, cent, cost

def get_distance(A, B):
    # simple matching
    dist = (A!=B).sum()
    return dist

def weighted_choice(choices):
    '''Given an iterator with (choice, weight) elements, randomly
    draws a choice with probabilities based on the weights.
    
    '''
    total = sum(w for c, w in choices)
    r = np.random.uniform(0, total)
    upto = 0
    for c, w in choices:
        if upto + w > r:
            return c
        upto += w
    assert False, "Shouldn't get this far. Check for zeros?"

def clustering_cost(X, clust, Xclust):
    '''Clustering cost, defined as the sum distance of all points
    to their respective clusters.
    
    '''
    cost = 0
    for ii in range(X.shape[0]):
        for jj in range(clust.shape[0]):
            if Xclust[ii] == jj:
                cost += get_distance(X[ii], clust[jj])
    return cost

if __name__ == "__main__":
    # reproduce results in Huang [1998]
    
    # load small soybean disease data set
    X = np.genfromtxt('/home/nico/Code/kaggle/Employee/soybean.csv', dtype='int64', delimiter=',')[:,:-1]
    y = np.genfromtxt('/home/nico/Code/kaggle/Employee/soybean.csv', dtype='unicode', delimiter=',', usecols=35)
    
    # drop columns with single value
    X = X[:,np.std(X, axis=0) > 0.]
    
    Xclust, cent, cost = opt_kmodes(X, 4, preruns=10, goodpctl=20, centUpd='wsampling', maxiters=40)
    
    classtable = np.zeros((4,4), dtype='int64')
    for ii,_ in enumerate(y):
        classtable[int(y[ii][-1])-1,Xclust[ii]] += 1
    
    print("    | Clust 1 | Clust 2 | Clust 3 | Clust 4 |")
    print("----|---------|---------|---------|---------|")
    for ii in range(4):
        prargs = tuple([ii+1] + list(classtable[ii,:]))
        print(" D{0} |      {1:>2} |      {2:>2} |      {3:>2} |      {4:>2} |".format(*prargs))
    
