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
    cent = np.empty((k, dim))
    
    # ----------------------
    # INIT [see Huang, 1998]
    # ----------------------
    
    # cluster that points belong to
    Xclust = np.zeros(N, dtype='int64')
    # determine frequencies of attributes, necessary for smart init
    freqs = []
    for idim in range(dim):
        bc = np.bincount(X[:,idim])
        inds = np.nonzero(bc)[0]
        bc = bc[inds]
        # sorted from most to least frequent
        sinds = np.argsort(bc)[::-1]
        freqs.append(zip(inds[sinds], bc[sinds]))
    for ik in range(k):
        for idim in range(dim):
            rd = np.random.randint(0,k)
            if rd < len(freqs[idim]):
                cent[ik, idim] = freqs[idim][rd][0]
            else:
                # otherwise sample centroids using the probabilities of attributes
                # (I assume that's what's meant in the Huang [1998] paper)
                cent[ik, idim] = weighted_choice(freqs[idim])
    
    # could result in empty clusters, so set centroid to closest point in X
    minInds = []
    for ik in range(k):
        minDist = np.inf
        for iN in range(N):
            # extra check here: we don't want 2 the same centroids
            if iN not in minInds:
                dist = get_distance(X[iN], cent[ik])
                if dist < minDist:
                    minDist = dist
                    minInd = iN
        minInds.append(minInd)
        cent[ik] = X[minInd]
    
    # ----------------------
    # ITERATION
    # ----------------------
    # we don't assume to know how many unique attributes there are for each point (last dimension),
    # so work with sparse matrix (dok is efficient for incremental filling, like we do here)
    # (note: can't use [...]*k, because copies are not allowed for dok)
    freqClust = []
    for ii in range(k):
        freqClust.append(sparse.dok_matrix((dim, dim)))
    
    iters = 0
    converged = False
    while iters <= maxiters and not converged:
        moves = 0
        for iN in range(N):
            minDist = np.inf
            for ik in range(k):
                dist = get_distance(X[iN], cent[ik])
                if dist < minDist:
                    minDist = dist
                    cluster = ik
            # assign to cluster and record initial frequencies of attributes
            if iters == 0:
                Xclust[iN] = cluster
                for idim in range(dim):
                    freqClust[cluster][idim, X[iN, idim]] += 1
            
            # if necessary, move point and update centroids
            if Xclust[iN] != cluster or iters == 1:
                moves += 1
                oldcluster = Xclust[iN]
                for idim in range(dim):
                    # update frequencies of attributes in clusters
                    freqClust[cluster][idim, X[iN, idim]] += 1
                    freqClust[oldcluster][idim, X[iN, idim]] -= 1
                    # update the centroids by choosing the most likely attribute
                    cent[cluster, idim] = np.argmax(freqClust[cluster][idim,:].todense())
                    cent[oldcluster, idim] = np.argmax(freqClust[oldcluster][idim,:].todense())
                Xclust[iN] = cluster
        converged = (moves == 0 and iters > 0)
        print("Iteration: %d, moves: %d" % (iters, moves))
        iters += 1
        
    return Xclust, cent

def opt_kmodes(**kwargs):
    '''Tries to ensure a good clustering result by choosing one that has a
    relatively low clustering cost compared to the costs of a number of pre-runs.
    '''
    precosts = []
    print("Starting preruns...")
    for ii in range(kwargs['preruns']):
        Xclust, cent = kmodes(kwargs['X'], kwargs['k'], kwargs['maxiters'])
        precosts.append(clustering_cost(kwargs['X'], cent, Xclust))
    
    while True:
        Xclust, cent = kmodes(kwargs['X'], kwargs['k'], kwargs['maxiters'])
        cost = clustering_cost(kwargs['X'], cent, Xclust)
        if cost <= np.percentile(precosts, kwargs['goodpctl']):
            print("Found a good clustering.")
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
    print(choices)
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
    X = np.genfromtxt('/home/nico/Code/kaggle/Employee/soybean.csv', dtype='int64', delimiter=',')[:,:-1]
    y = np.genfromtxt('/home/nico/Code/kaggle/Employee/soybean.csv', dtype='unicode', delimiter=',', usecols=35)
    
    useful = (np.std(X, axis=0) > 0.)
    #Xclust, cent = kmodes(X=X[:,useful], k=4, maxiters=40)
    Xclust, cent = opt_kmodes(X=X[:,useful], k=4, maxiters=40, preruns=10, goodpctl=20)
    
    classtable = np.zeros((4,4), dtype='int64')
    for ii,_ in enumerate(y):
        classtable[int(y[ii][-1])-1,Xclust[ii]] += 1
    
    print('     | Clust 1 | Clust 2 | Clust 3 | Clust 4 |')
    print('----------------------------------------------')
    for ii in range(4):
        print(' D{0}  |       {1} |       {2} |       {3} |       {4} |'.format(ii+1, \
                                                                         classtable[ii,0], \
                                                                         classtable[ii,1], \
                                                                         classtable[ii,2], \
                                                                         classtable[ii,3]))
    
