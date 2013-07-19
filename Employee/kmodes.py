#!/usr/bin/env python

import random
import numpy as np
from scipy.sparse import issparse

def kmodes(X, centroids, maxiter=10):
    '''out:
        centres, k x dim
        Xtocentre: each X -> its nearest centre, ints N -> k
        distances, N
    '''
    
    N, dim = X.shape
    k, cdim = centroids.shape
    assert dim == cdim
    
    allx = np.arange(N)
    for jiter in range( 1, maxiter+1 ):
        D = get_distances(X, centroids)
        xtoc = D.argmin(axis=1)
        Xdistances = D[allx,xtoc]
        print "Median distance X to nearest centroid = %.4g" % Xdistances.median()
        if jiter == maxiter:
            break
        for jc in range(k):
            c = np.where(xtoc==jc)[0]
            if len(c) > 0:
                nc = X[c].count_nonzero(axis=0)
                nc = X[xtoc].count_nonzero(axis=0)
                n = len(X)
                centroids[jc] = ( (nq/n) >= (nc/n) )
    print("%d iterations,  cluster sizes:" % jiter, np.bincount(xtoc))
    if verbose >= 2:
        r50 = np.zeros(k)
        r90 = np.zeros(k)
        for j in range(k):
            dist = distances[ xtoc == j ]
            if len(dist) > 0:
                r50[j], r90[j] = np.percentile( dist, (50, 90) )
        print "Cluster 50% radius", r50.astype(int)
        print "Cluster 90% radius", r90.astype(int)
            # scale L1 / dim, L2 / sqrt(dim) ?
    return centroids, xtoc, distances

#...............................................................................
def kmodesinit( X, k, nsample=0, **kwargs ):
    ''' 2-pass kmeans, fast for large N:
        1) kmeans a random sample of nsample ~ sqrt(N) from X
        2) full kmeans, starting from those centres
    '''
        # merge w kmeans ? mttiw
        # v large N: sample N^1/2, N^1/2 of that
        # seed like sklearn ?
    N, dim = X.shape
    if nsample == 0:
        nsample = max( 2*np.sqrt(N), 10*k )
    Xsample = randomsample( X, int(nsample) )
    pass1centres = randomsample( X, int(k) )
    samplecentres = kmeans( Xsample, pass1centres, **kwargs )[0]
    return kmeans( X, samplecentres, **kwargs )

def get_distances(X, C):
    D = np.array(X.shape[0], C.shape[0])
    for ii in X.shape[0]:
        for jj in C.shape[0]:
            D[ii,jj] = (X[ii,:] != C[jj,:]).sum()
    return D

def randomsample( X, n ):
    ''' random.sample of the rows of X
        X may be sparse -- best csr
    '''
    sampleix = random.sample( xrange( X.shape[0] ), int(n) )
    return X[sampleix]

def nearestcentres( X, centres, metric="euclidean", p=2 ):
    ''' each X -> nearest centre, any metric
            euclidean2 (~ withinss) is more sensitive to outliers,
            cityblock (manhattan, L1) less sensitive
    '''
    D = cdist( X, centres, metric=metric, p=p )  # |X| x |centres|
    return D.argmin(axis=1)

def Lqmetric( x, y=None, q=.5 ):
    # yes a metric, may increase weight of near matches; see ...
    return (np.abs(x - y) ** q) .mean() if y is not None \
        else (np.abs(x) ** q) .mean()

#...............................................................................
class Kmodes:
    ''' km = Kmodes( X, k= or centroids=, ... )
        in: either initial centroids= for kmodes
            or k= [nsample=] for kmodesinit
        out: km.centroids, km.Xtocentre, km.distances
        iterator:
            for jcentroid, J in km:
                clustercentroid = centroids[jcentroid]
                J indexes e.g. X[J], classes[J]
    '''
    def __init__( self, X, k=0, centroids=None, nsample=0, **kwargs ):
        self.X = X
        if centroids is None:
            self.centroids, self.Xtocentroid, self.distances = kmodesinit(
                X, k=k, nsample=nsample, **kwargs )
        else:
            self.centroids, self.Xtocentroid, self.distances = kmodes(
                X, centroids, **kwargs )

    def __iter__(self):
        for jc in range(len(self.centroids)):
            yield jc, (self.Xtocentroid == jc)

#...............................................................................
if __name__ == "__main__":
    import random
    import sys
    from time import time

    N = 10000
    dim = 10
    ncluster = 10
    kmsample = 100  # 0: random centres, > 0: kmeanssample
    kmdelta = .001
    kmiter = 10
    metric = "cityblock"  # "chebyshev" = max, "cityblock" L1,  Lqmetric
    seed = 1

    exec( "\n".join( sys.argv[1:] ))  # run this.py N= ...
    np.set_printoptions( 1, threshold=200, edgeitems=5, suppress=True )
    np.random.seed(seed)
    random.seed(seed)

    print "N %d  dim %d  ncluster %d  kmsample %d  metric %s" % (
        N, dim, ncluster, kmsample, metric)
    X = np.random.exponential( size=(N,dim) )
        # cf scikits-learn datasets/
    t0 = time()
    if kmsample > 0:
        centres, xtoc, dist = kmeanssample( X, ncluster, nsample=kmsample,
            delta=kmdelta, maxiter=kmiter, metric=metric, verbose=2 )
    else:
        randomcentres = randomsample( X, ncluster )
        centres, xtoc, dist = kmeans( X, randomcentres,
            delta=kmdelta, maxiter=kmiter, metric=metric, verbose=2 )
    print "%.0f msec" % ((time() - t0) * 1000)
