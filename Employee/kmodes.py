#!/usr/bin/env python

'''
Implementation of the k-modes algorithm by Huang [1997, 1998].
Uses the initialization scheme suggested in the latter paper.

N.J. de Vos
'''

from scipy import sparse
import numpy as np

def kmodes(X, k, maxiters):
    # input: data (no. points * no. attributes), number of clusters
    # returns: cluster numbers for elements of X, centroids
    
    # convert to numpy array, if needed
    X = np.asanyarray(X)
    N, dim = X.shape
    cent = np.array(k, dim)
    
    # ----------------------
    # INIT [see Huang, 1998]
    # ----------------------
    
    # cluster that points belong to
    Xclust = np.zeros(N)
    # determine frequencies of attributes, necessary for smart init
    freqs = [0]*len(dim)
    for idim in range(dim):
        bc = np.bincount(X[:,idim])
        inds = np.nonzero(bc)[0] # delete [0]?
        # sorted from most to least frequent
        sinds = np.argsort(inds)[::-1]
        freqs[idim] = zip(inds[sinds], bc[sinds])
    # now sample centroids using the probabilities of attributes
    # (I assume that's what's meant in the Huang [1998] paper)
    for ik in range(k):
        for idim in range(dim):
            cent[ik, idim] = weighted_choice(freqs[idim])
        # could result in empty clusters, so set centroid to closest point in X
        minDist = np.inf
        for iN in range(N):
            dist = get_distance(X[iN], cent[ik])
            # extra check here: we don't want 2 the same centroids
            if dist < minDist and not (X[iN]==cent[:ik]).any():
                minDist = dist
                minInd = iN
        cent[ik] = X[minInd]
    
    # ----------------------
    # ITERATION
    # ----------------------
    # we don't assume to know how many unique attributes there are for each point (last dimension),
    # so work with sparse matrix (dok is efficient for incremental filling, like we do here)
    freqClust = sparse.dok_matrix((k, dim, dim))
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
            # record initial frequencies of attributes in clusters
            if iter == 1:
                for idim in range(dim):
                freqClust[cluster, idim, X[iN, idim]] += 1
            
            # if necessary, move point and update centroids
            if Xclust[iN] ne cluster:
                moves += 1
                oldcluster = Xclust[iN]
                for idim in range(dim):
                    # update frequencies of attributes in clusters
                    freqClust[cluster, idim, X[iN, idim]] += 1
                    freqClust[oldcluster, idim, X[iN, idim]] -= 1
                    # update the centroids by choosing the most likely attribute
                    cent[cluster, idim] = np.amax(freqClust[cluster, idim])
                    cent[oldcluster, idim] = np.amax(freqClust[oldcluster, idim])
        converged = (moves == 0)
        
    return Xclust, cent

def opt_kmodes(*args, preruns=10, goodpctl=20):
    '''Tries to ensure a good clustering result by choosing one that has a
    relatively low clustering cost compared to the costs of a number of pre-runs.
    '''
    precosts = []
    for ii in range(preruns):
        Xclust, cent = kmodes(args)
        precosts.append(clustering_cost(X, cent, Xclust))
    
    while True:
        Xclust, cent = kmodes(args)
        cost = clustering_cost(X, cent, Xclust)
        if cost <= np.percentile(ccosts, goodpctl):
            break
    
    return Xclust, cent

def get_distance(A, B):
    # simple matching
    return (A!=B).sum()

def weighted_choice(choices):
    total = sum(w for c, w in choices)
    r = np.random.uniform(0, total)
    upto = 0
    for c, w in choices:
        if upto + w > r:
            return c
        upto += w
    # shouldn't get this far
    assert False

def clustering_cost(X, clust, Xclust):
    cost = 0
    for ii in range(X.shape[0]):
        for jj in range(clust.shape[0]):
            if Xclust[ii] == jj:
                cost += get_distance(X[ii], clust[jj])
    return cost

if __name__ == "__main__":
    # load soybean disease data
    X = np.recfromcsv('soybean.csv')
    y = X[:,-1]
    X = X[:,:-1]
    
    useful = (np.std(X, axis=0) > 0)
    Xclust, cent = opt_kmodes(X[useful], 4, maxiters=1000)
    
    classtable = sparse.dok_matrix((4,4))
    for ii in len(y):
        classtable[ii,Xclust[ii]] += 1
    
    print('    | Clust 1 | Clust 2 | Clust 3 | Clust 4 |')
    print('-----------------------------------------------------')
    for ii in range(4):
        print(' {0} |       {1} |       {2} |      {3} |        {4} |'.format(ii+1, classtable[ii,:]))
    
