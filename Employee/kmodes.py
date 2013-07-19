#!/usr/bin/env python

'''
Simple implementation of the k-modes algorithm by Huang [1997, 1998]

N.J. de Vos
'''

import random
import numpy as np

def kmodes(X, k, maxiters=10):
    # input: data (no. points * no. attributes), number of clusters
    # returns: cluster numbers for elements of X, centroids
    
    # convert to numpy array, if needed
    X = np.asanyarray(X)
    N, dim = X.shape
    cent = np.array(k, dim)
    
    # ----------------------
    # INIT [see Huang, 1998]
    # ----------------------
    
    Xclust = np.zeros(N)
    # determine frequencies of attributes
    freqs = [0]*len(dim)
    for idim in range(dim):
        bc = np.bincount(X[:,idim])
        inds = np.nonzero(bc)[0] # delete [0]?
        sinds = np.argsort(inds)[::-1]
        freqs[idim] = zip(inds[sinds], bc[sinds])
    # now sample centroids using the probabilities of attributes
    # (I assume that's what's meant in the paper)
    for ik in range(k):
        for idim in range(dim):
            cent[ik, idim] = weighted_choice(freqs[idim])
        # could result in empty clusters, so set centroid to closest point in X
        minDist = 1e12
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
    # (dim, dim), because last one might have all unique attributes
    freqClust = np.zeros(k, dim, dim)
    iters = 0
    converged = False
    while iters <= maxiters and not converged:
        iters += 1
        moves = 0
        for iN in range(N):
            minDist = 1e12
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
                    cent[cluster, idim] = freqClust[cluster, idim].argmax()
                    cent[oldcluster, idim] = freqClust[oldcluster, idim].argmax()
        converged = (moves == 0)
        
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

if __name__ == "__main__":
    pass
