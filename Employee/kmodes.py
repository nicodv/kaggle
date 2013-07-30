#!/usr/bin/env python

'''
Implementation of traditional and fuzzy k-modes clustering algorithms.
'''
__author__  = 'Nico de Vos'
__email__   = 'njdevos@gmail.com'
__license__ = 'MIT'
__version__ = '0.4'

import random
import numpy as np
from collections import defaultdict


class KModes(object):
    
    def __init__(self, k):
        '''k-modes clustering algorithm for categorical data.
        See Huang [1997, 1998] or Chaturvedi et al. [2001].
        
        Inputs:     k       = number of clusters
        Attributes: Xclust  = cluster numbers [no. points]
                    cent    = centroids [k * no. attributes]
                    cost    = clustering cost, defined as the sum distance of
                              all points to their respective clusters
        
        '''
        assert k > 1, "Choose at least 2 clusters."
        self.k = k
    
    def cluster(self, X, init='Huang', maxIters=100, verbose=1):
        '''Inputs:  X           = data points [no. attributes * no. points]
                    init        = initialization method ('Huang' for the one described in
                                  Huang [1998], 'Cao' for the one in Cao et al. [2009])
                    maxIters    = maximum no. of iterations
        '''
        # convert to numpy array, if needed
        X = np.asanyarray(X)
        N, at = X.shape
        assert self.k < N, "More clusters than data points?"
        
        self.init = init
        
        # ----------------------
        #    INIT
        # ----------------------
        print("Init: initializing centroids")
        cent = self.init_centroids(X)
        
        print("Init: initializing clusters")
        member = np.zeros((self.k, N), dtype='int32')
        # clustFreq is a list of lists with dictionaries that contain the
        # frequencies of values per cluster and attribute
        clustFreq = [[defaultdict(int) for _ in range(at)] for _ in range(self.k)]
        for ix, curx in enumerate(X):
            # initial assigns to clusters
            dissim = self.get_dissim(cent, curx)
            cluster = np.argmin(dissim)
            member[cluster,ix] = 1
            # count attribute values per cluster
            for iat, val in enumerate(curx):
                clustFreq[cluster][iat][val] += 1
        # perform an initial centroid update
        for ik in range(self.k):
            for iat in range(at):
                cent[ik,iat] = key_for_max_value(clustFreq[ik][iat])
        
        # ----------------------
        #    ITERATION
        # ----------------------
        print("Starting iterations...")
        itr = 0
        converged = False
        while itr <= maxIters and not converged:
            itr += 1
            moves = 0
            for ix, curx in enumerate(X):
                dissim = self.get_dissim(cent, curx)
                cluster = np.argmin(dissim)
                # if necessary: move point, and update old/new cluster frequencies and centroids
                if not member[cluster, ix]:
                    moves += 1
                    oldcluster = np.argwhere(member[:,ix])
                    member[oldcluster,ix] = 0
                    member[cluster,ix] = 1
                    for iat, val in enumerate(curx):
                        # update frequencies of attributes in clusters
                        clustFreq[cluster][iat][val] += 1
                        clustFreq[oldcluster][iat][val] -= 1
                        # update new and old centroids by choosing most likely attribute
                        for curc in (cluster, oldcluster):
                            cent[curc, iat] = key_for_max_value(clustFreq[curc][iat])
                    if verbose == 2:
                        print("Move from cluster {0} to {1}".format(oldcluster, cluster))
            
            # all points seen in this iteration
            converged = (moves == 0)
            if verbose:
                print("Iteration: {0}/{1}, moves: {2}".format(itr, maxIters, moves))
        
        self.cost = self.clustering_cost(X, cent, member)
        self.cent = cent
        self.Xclust = np.array([int(np.argwhere(member[:,x])) for x in range(N)])
        self.member = member
    
    def init_centroids(self, X):
        assert self.init in ('Huang', 'Cao')
        N, at = X.shape
        cent = np.empty((self.k, at))
        if self.init == 'Huang':
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
                for ik in range(self.k):
                    cent[ik, iat] = random.choice(choices)
            # the previously chosen centroids could result in empty clusters,
            # so set centroid to closest point in X
            for ik in range(self.k):
                dissim = self.get_dissim(X, cent[ik])
                ndx = dissim.argsort()
                # and we want the centroid to be unique
                while np.all(X[ndx[0]] == cent, axis=1).any():
                    ndx = np.delete(ndx, 0)
                cent[ik] = X[ndx[0]]
        elif self.init == 'Cao':
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
            dissim = self.get_dissim(X, cent[0])
            cent[1] = X[np.argmax(dissim * dens)]
            # for the reamining centroids, choose max dens * dissim to the (already assigned)
            # centroid with the lowest dens * dissim
            for ic in range(2,self.k):
                dd = np.empty((ic, N))
                for icp in range(ic):
                    dd[icp] = self.get_dissim(X, cent[icp]) * dens
                cent[ic] = X[np.argmax(np.min(dd, axis=0))]
        
        return cent
    
    def get_dissim(self, A, b):
        # TODO: add other dissimilarity measures?
        # simple matching dissimilarity
        return (A != b).sum(axis=1)
    
    def clustering_cost(self, X, cent, member, alpha=1):
        # generalized form with alpha. alpha > 1 for fuzzy k-modes
        cost = 0
        for iN, curx in enumerate(X):
            cost += np.sum( self.get_dissim(cent, curx) * (member[:,iN] ** self.alpha) )
        return cost

################################################################################

class FuzzyKModes(KModes):
    
    def __init__(self, k, alpha=1.5):
        '''Fuzzy k-modes clustering algorithm for categorical data.
        Uses traditional, hard centroids, following Huang and Ng [1999].
        
        Inputs:     k           = number of clusters
                    alpha       = alpha coefficient
        Attributes: Xclust  = cluster numbers with max. membership [no. points]
                    member  = membership matrix [k * no. points]
                    cent    = centroids [k * no. attributes]
                    cost    = clustering cost
        
        '''
        super(FuzzyKModes, self).__init__(k)
        
        assert alpha > 1, "alpha should be > 1 (alpha = 1 equals regular k-modes)."
        self.alpha = alpha
    
    def cluster(self, X, init='Huang', maxIters=200, costInter=5, verbose=1):
        '''Inputs:  X           = data points [no. attributes * no. points]
                    init        = initialization method ('Huang' for the one described in
                                  Huang [1998], 'Cao' for the one in Cao et al. [2009]).
                    maxIters    = maximum no. of iterations
                    costInter   = frequency with which to check the total cost
                                  (for speeding things up, since it is computationally expensive)
        
        '''
        
        # convert to numpy array, if needed
        X = np.asanyarray(X)
        N, at = X.shape
        assert self.k < N, "More clusters than data points?"
        
        # ----------------------
        #    INIT
        # ----------------------
        print("Init: initializing centroids")
        cent = self.init_centroids(X)
        
        # store for all attributes which points have a certain attribute value
        domAtX = [defaultdict(list) for _ in range(at)]
        for iN, curx in enumerate(X):
            for iat, val in enumerate(curx):
                domAtX[iat][val].append(iN)
        
        # ----------------------
        #    ITERATION
        # ----------------------
        print("Starting iterations...")
        itr = 0
        converged = False
        lastCost = np.inf
        while itr <= maxIters and not converged:
            member = self.update_membership(cent, X)
            for ik in range(self.k):
                cent[ik] = self.update_centroid(domAtX, member[ik])
            
            # computationally expensive, only check every N steps
            if itr % costInter == 0:
                cost = self.clustering_cost(X, cent, member)
                converged = cost >= lastCost
                lastCost = cost
                if verbose:
                    print("Iteration: {0}/{1}, cost: {2}".format(itr, maxIters, cost))
            itr += 1
        
        self.cost = cost
        self.cent = cent
        self.Xclust = np.array([int(np.argmax(member[:,x])) for x in range(N)])
        self.member = member
    
    def update_membership(self, cent, X):
        N = X.shape[0]
        member = np.empty((self.k, N))
        for iN, curx in enumerate(X):
            dissim = self.get_dissim(cent, curx)
            if np.any(dissim == 0):
                member[:,iN] = np.where(dissim == 0, 1, 0)
            else:
                for ik, curc in enumerate(cent):
                    factor = 1. / (self.alpha - 1)
                    member[ik,iN] = 1 / np.sum( (float(dissim[ik]) / dissim)**factor )
        return member
    
    def update_centroid(self, domAtX, member):
        cent = []
        for iat in range(len(domAtX)):
            # return attribute that maximizes the sum of the memberships
            v = list(domAtX[iat].values())
            k = list(domAtX[iat].keys())
            memvar = [sum(member[x]**self.alpha) for x in v]
            cent.append(k[np.argmax(memvar)])
        return np.array(cent)

################################################################################

class FuzzyFuzzyKModes(KModes):
    
    def __init__(self, k, alpha=1.5):
        '''Fuzzy k-modes clustering algorithm for categorical data.
        Uses fuzzy centroids, following and Kim et al. [2004].
        
        Inputs:     k           = number of clusters
                    alpha       = alpha coefficient
        Attributes: Xclust  = cluster numbers with max. membership [no. points]
                    member  = membership matrix [k * no. points]
                    omega   = fuzzy centroids
                    cost    = clustering cost
        
        '''
        super(FuzzyFuzzyKModes, self).__init__(k)
        
        assert alpha > 1, "alpha should be > 1 (alpha = 1 equals regular k-modes)."
        self.alpha = alpha
    
    def cluster(self, X, maxIters=200, costInter=5, verbose=1):
        '''Inputs:  X           = data points [no. attributes * no. points]
                    maxIters    = maximum no. of iterations
                    costInter   = frequency with which to check the total cost
                                  (for speeding things up, since it is computationally expensive)
        
        '''
        
        # convert to numpy array, if needed
        X = np.asanyarray(X)
        N, at = X.shape
        assert self.k < N, "More clusters than data points?"
        
        # ----------------------
        #    INIT
        # ----------------------
        # count all attributes
        freqAt = [defaultdict(int) for _ in range(at)]
        for curx in X:
            for iat, val in enumerate(curx):
                freqAt[iat][val] += 1
        
        # omega = fuzzy set (as dict) for each attribute per cluster
        omega = [[{} for _ in range(at)] for _ in range(self.k)]
        for ik in range(self.k):
            for iat in range(at):
                rands = np.random.rand(len(freqAt[iat]))
                # to make sum omegas = 1 per attribute
                rands = rands / sum(rands)
                for ival, val in enumerate(freqAt[iat]):
                    omega[ik][iat][val] = rands[ival]
        
        # ----------------------
        #    ITERATION
        # ----------------------
        print("Starting iterations...")
        itr = 0
        converged = False
        lastCost = np.inf
        while itr <= maxIters and not converged:
            # O(k*N*at*no. of unique values)
            member = self.update_membership(X, omega)
            # O(k*N*at)
            for ik in range(self.k):
                omega[ik] = self.update_centroid(X, member[ik])
            
            # computationally expensive, only check every N steps
            if itr % costInter == 0:
                cost = self.clustering_cost(X, member, omega)
                converged = cost >= lastCost
                lastCost = cost
                if verbose:
                    print("Iteration: {0}/{1}, cost: {2}".format(itr, maxIters, cost))
            itr += 1
        
        self.cost = cost
        self.omega = omega
        self.Xclust = np.array([int(np.argmax(member[:,x])) for x in range(N)])
        self.member = member
    
    def update_membership(self, X, omega):
        N = X.shape[0]
        member = np.empty((self.k, N))
        for iN, curx in enumerate(X):
            dissim = self.get_fuzzy_dissim(omega, curx)
            if np.any(dissim == 0):
                member[:,iN] = np.where(dissim == 0, 1, 0)
            else:
                for ik, curc in enumerate(omega):
                    factor = 1. / (self.alpha - 1)
                    member[ik,iN] = 1 / np.sum( (float(dissim[ik]) / dissim)**factor )
        return member
    
    def update_centroid(self, X, member):
        omega = [defaultdict(float) for _ in range(X.shape[1])]
        for iN, curx in enumerate(X):
            for iat, val in enumerate(curx):
                omega[iat][val] += member[iN] ** self.alpha
        return omega
    
    def get_fuzzy_dissim(self, omega, x):
        dist = np.zeros(len(omega))
        for ik in curo in enumerate(omega):
            for iat, val in enumerate(curo):
                dist[ik] += sum( np.where(val == b[iat], 0, omega[ik][iat][val]) )
        return dist
    
    def clustering_cost(self, X, omega, member):
        cost = 0
        for iN, curx in enumerate(X):
            cost += np.sum( self.get_fuzzy_dissim(omega, curx) * (member[:,iN] ** self.alpha) )
        return cost


def key_for_max_value(d):
    #Very fast method (supposedly) to get key for maximum value in dict.
    v = list(d.values())
    k = list(d.keys())
    return k[v.index(max(v))]

def opt_kmodes(k, X, preRuns=10, goodPctl=20, **kwargs):
    '''Shell around k-modes algorithm that tries to ensure a good clustering result
    by choosing one that has a relatively low clustering cost compared to the
    costs of a number of pre-runs. (Huang [1998] states that clustering cost can be
    used to judge the clustering quality.)
    
    Returns a (good) KModes class instantiation.
    
    '''
    
    if kwargs['init'] == 'Cao' and kwargs['centUpd'] == 'mode':
        print("Hint: Cao initialization method + mode updates = deterministic. \
                No opt_kmodes necessary, run kmodes method directly instead.")
    
    preCosts = []
    print("Starting preruns...")
    for _ in range(preRuns):
        kmodes = KModes(k)
        kmodes.cluster(X, verbose=0, **kwargs)
        preCosts.append(kmodes.cost)
        print("Cost = {0}".format(kmodes.cost))
    
    while True:
        kmodes = KModes(k)
        kmodes.cluster(X, verbose=1, **kwargs)
        if kmodes.cost <= np.percentile(preCosts, goodPctl):
            print("Found a good clustering.")
            print("Cost = {0}".format(kmodes.cost))
            break
    
    return kmodes

if __name__ == "__main__":
    # reproduce results on small soybean data set
    X = np.genfromtxt('/home/nico/Code/kaggle/Employee/soybean.csv', dtype='int64', delimiter=',')[:,:-1]
    y = np.genfromtxt('/home/nico/Code/kaggle/Employee/soybean.csv', dtype='unicode', delimiter=',', usecols=35)
    
    # drop columns with single value
    X = X[:,np.std(X, axis=0) > 0.]
    
    #kmodes_h = opt_kmodes(4, X, preRuns=10, goodPctl=20, init='Huang', maxIters=100)
    kmodes_huang = KModes(4)
    kmodes_huang.cluster(X, init='Huang')
    kmodes_cao = KModes(4)
    kmodes_cao.cluster(X, init='Cao')
    fkmodes = FuzzyKModes(4, alpha=1.1)
    fkmodes.cluster(X)
    ffkmodes = FuzzyFuzzyKModes(4, alpha=1.8)
    ffkmodes.cluster(X)
    
    for result in (kmodes_huang, kmodes_cao, fkmodes, ffkmodes):
        classtable = np.zeros((4,4), dtype='int64')
        for ii,_ in enumerate(y):
            classtable[int(y[ii][-1])-1,result.Xclust[ii]] += 1
        
        print("    | Clust 1 | Clust 2 | Clust 3 | Clust 4 |")
        print("----|---------|---------|---------|---------|")
        for ii in range(4):
            prargs = tuple([ii+1] + list(classtable[ii,:]))
            print(" D{0} |      {1:>2} |      {2:>2} |      {3:>2} |      {4:>2} |".format(*prargs))
    
