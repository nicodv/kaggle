#!/usr/bin/env python

'''
Implementation of the k-modes clustering algorithm.
'''
__author__  = 'Nico de Vos'
__email__   = 'njdevos@gmail.com'
__license__ = 'MIT'
__version__ = '0.3'

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
                    cost    = clustering cost
        
        '''
        assert k > 1, "Choose at least 2 clusters."
        self.k = k
    
    def cluster(self, X, init='Cao', centUpd='mode', maxIters=100, verbose=1):
        '''Inputs:  X           = data points [no. attributes * no. points]
                    init        = initialization method ('Huang' for the one described in
                                  Huang [1998], 'Cao' for the more advanced one in
                                  Cao et al. [2009])
                    centUpd     = centroid updating method ('mode' for most common
                                  attribute or 'wsample' for weighted sampling)
                    maxIters    = maximum no. of iterations
        '''
        # convert to numpy array, if needed
        X = np.asanyarray(X)
        N, at = X.shape
        assert self.k < N, "More clusters than data points?"
        
        self.init = init
        self.centUpd = centUpd
        
        # ----------------------
        #    INIT
        # ----------------------
        print("Init: initializing centroids")
        cent = _init_centroids(X)
        
        print("Init: initializing clusters")
        member = np.zeros((self.k, N))
        # clustFreq is a list of lists with dictionaries that contain the
        # frequencies of values per cluster and attribute
        clustFreq = [[defaultdict(int) for _ in range(at)] for _ in range(self.k)]
        for ix, curx in enumerate(X):
            # initial assigns to clusters
            dissim = _get_dissim(cent, curx)
            cluster = dissim.argsort()[0]
            member[cluster,ix] = 1
            # count attribute values per cluster
            for iat, val in enumerate(curx):
                clustFreq[cluster][iat][val] += 1
        # perform an initial centroid update
        for ik in range(self.k):
            for iat in range(at):
                cent[ik,iat] = _update_centroid(clustFreq[ik][iat])
        
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
                dissim = get_dissim(cent, curx)
                cluster = dissim.argsort()[0]
                # if necessary: move point, and update old/new cluster frequencies and centroids
                if not member[cluster, ix]:
                    moves += 1
                    oldcluster = np.argwhere(member[:,ix])
                    member[oldcluster,ix] = 0
                    member[cluster,ix] = 1
                    assert all(np.sum(member, axis=0) == 1)
                    assert all(0 < np.sum(member, axis=1) < N)
                    for iat, val in enumerate(curx):
                        # update frequencies of attributes in clusters
                        clustFreq[cluster][iat][val] += 1
                        clustFreq[oldcluster][iat][val] -= 1
                        assert clustFreq[oldcluster][iat][val] >= 0
                        # update the new and old centroids by choosing (from) the most likely attribute(s)
                        for curc in (cluster, oldcluster):
                            cent[curc, iat] = _update_centroid(clustFreq[curc][iat])
                    if verbose == 2:
                        print("Move from cluster {0} to {1}".format(oldcluster, cluster))
            
            # all points seen in this iteration
            converged = (moves == 0)
            if verbose:
                print("Iteration: {0}/{1}, moves: {2}".format(itr, maxIters, moves))
        
        self.cost = clustering_cost(X, cent, member, alpha=1)
        self.cent = cent
        self.Xclust = np.array([np.argwhere(member[x]) for x in range(N)])
    
    def _init_centroids(self, X):
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
                dissim = get_dissim(X, cent[ik])
                ndx = dissim.argsort()
                # we want the centroid to be unique
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
            dissim = get_dissim(X, cent[0])
            cent[1] = X[np.argmax(dissim * dens)]
            # for the reamining centroids, choose max dens * dissim to the (already assigned)
            # centroid with the lowest dens * dissim
            for ic in range(2,self.k):
                dd = np.empty((ic, N))
                for icp in range(ic):
                    dd[icp] = get_dissim(X, cent[icp]) * dens
                cent[ic] = X[np.argmax(np.min(dd, axis=0))]
        
        return cent
    
    def _update_centroid(self,freqs):
        # TODO: Taking the mode (i.e. highest frequency element), which is what Huang [1998]
        # suggests, converges faster and takes less processing power. However, weighted sampling
        # in the centroid update makes intuitive sense to me and might offer better
        # generalization. Investigate.
        assert self.centUpd in ('mode', 'wsample')
        if self.centUpd == 'mode':
            return key_for_max_value(freqs)
        elif self.centUpd == 'wsample':
            choices = [chc for chc, wght in freqs.items() for _ in range(wght)]
            return random.choice(choices)
    
    def get_dissim(self, A, b):
        # TODO: add other dissimilarity measures?
        # simple matching dissimilarity
        return (A != b).sum(axis=1)
    
    def clustering_cost(self, X, clust, member):
        '''Clustering cost, defined as the sum distance of all points
        to their respective clusters.
        
        '''
        cost = 0
        for ic, curc in enumerate(clust):
            cost += get_dissim(X[member[ic]], curc).sum()
        return cost

################################################################################

class FuzzyKModes(KModes):
    
    def __init__(self, k, alpha):
        '''Fuzzy k-modes clustering algorithm for categorical data.
        See Huang and Ng [1999] and Kim et al. [2004].
        
        Inputs:     k       = number of clusters
                    alpha   = alpha coefficient
        Attributes: member  = membership matrix [k * no. points]
                    cent    = centroids [k * no. attributes]
                    cost    = clustering cost
        
        '''
        super(FuzzyKModes).__init__(self, k)
        
        assert alpha > 1, "alpha should be > 1 (alpha = 1 equals regular k-modes)."
        self.alpha = alpha
    
    def cluster(self, X, init='Cao', centType='fuzzy', maxIters=100, verbose=1):
        '''Inputs:  X           = data points [no. attributes * no. points]
                    init        = initialization method ('Huang' for the one described in
                                  Huang [1998], 'Cao' for the more advanced one in
                                  Cao et al. [2009]). In case of fuzzy centroids,
                                  an additional fuzzification is performed.
                    centType    = centroid type ('hard' for traditional, hard
                                  centroids [Huang and Ng, 1999] or 'fuzzy' for
                                  fuzzy centroids [Kim et al., 2004])
                    maxIters    = maximum no. of iterations
        
        '''
        
        self.init = init
        self.centType = centType
        
        # ----------------------
        #    INIT
        # ----------------------
        print("Init: initializing centroids")
        cent = _init_centroids(X)
        if self.centType == 'fuzzy':
            cent = _fuzzify_centroids(cent)
        
        # store for all attributes which points have a certain attribute value
        # this is the main input to the centroid update
        domAtX = [defaultdict(list) for _ in range(at)]
        for iN, curx in enumerate(X):
            for iat, at in enumerate(curx):
                domAtX[iat][at].append(iat)
        
        # ----------------------
        #    ITERATION
        # ----------------------
        print("Starting iterations...")
        itr = 0
        tiny = 1e-6
        converged = False
        lastCost = np.inf
        while itr <= maxIters and not converged:
            member = _update_fuzzy_membership(cent, X)
            assert all(1-tiny < np.sum(member, axis=0) < 1+tiny)
            assert all(0 < np.sum(member, axis=1) < N)
            for ik in range(k):
                for iat in range(at):
                    cent[ik][iat] = _update_centroid(domAtX[iat], member[ik])
            cost = clustering_cost(X, cent, member)
            converged = cost == lastCost
            lastCost = cost
            if verbose:
                print("Iteration: {0}/{1}, cost: {2}".format(itr, maxIters, cost))
            itr += 1
    
    def _fuzzify_centroids(self, cent):
        pass
    
    def _update_fuzzy_membership(self, cent, X):
        k = cent.shape[0]
        N = X.shape[0]
        member = np.empty((self.k, N))
        for iN, curx in enumerate(X):
            if self.centType == 'crisp':
                dissim = get_dissim(cent, curx)
            elif self.centType == 'fuzzy':
                dissim = get_fuzzy_dissim(cent, curx)
            if np.any(dissim == 0):
                member[:,iN] = np.where(dissim == 0, 1, 0)
            else:
                for ik, curc in enumerate(cent):
                    num = dissim[ik]
                    denom = np.delete(dissim,ik)
                    member[ik,iN] = 1 / np.sum( (num / denom)**(1/(self.alpha-1)) )
        return member
    
    def _update_centroid(self, domAtX, member, centType):
        assert centType in ('hard','fuzzy')
        if centType == 'hard':
            # return attribute that maximizes the sum of the memberships
            v = list(domAtX.values())
            k = list(domAtX.keys())
            return k[v.index(max(sum(member(v)**self.alpha)))]
        elif centType == 'fuzzy':
            pass
        return upd
    
    def _get_fuzzy_dissim(self, A, b):
        pass


def key_for_max_value(d):
    #Very fast method (supposedly) to get key for maximum value in dict.
    v = list(d.values())
    k = list(d.keys())
    return k[v.index(max(v))]

def opt_kmodes(X, k, preRuns=10, goodPctl=20, **kwargs):
    '''Shell around k-modes algorithm that tries to ensure a good clustering result
    by choosing one that has a relatively low clustering cost compared to the
    costs of a number of pre-runs. (Huang [1998] states that clustering cost can be
    used to judge the clustering quality.)
    
    Returns a (good) KModes class instantiation.
    
    '''
    
    if kwargs['init'] == 'Cao' and kwargs['centUpd'] == 'mode':
        print("""Hint: Cao initialization method + mode updates = deterministic.
                No opt_kmodes necessary, run kmodes method directly instead.""")
    
    preCosts = []
    print("Starting preruns...")
    for _ in range(preRuns):
        kmodes = KModes(k).cluster(X, verbose=0, **kwargs)
        preCosts.append(kmodes.cost)
        print("Cost = {0}".format(kmodes.cost))
    
    while True:
        kmodes = KModes(k).cluster(X, verbose=1, **kwargs)
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
    
    #kmodes = opt_kmodes(X, 4, preRuns=10, goodPctl=20, init='Huang', centUpd='mode', maxIters=100)
    kmodes = KModes(4).cluster(X, init='Huang', centUpd='mode', maxIters=100)
    fkmodes = FuzzyKModes(4).cluster(X, init='Cao', centType='hard', maxIters=100)
    
    classtable = np.zeros((4,4), dtype='int64')
    for ii,_ in enumerate(y):
        classtable[int(y[ii][-1])-1,kmodes.Xclust[ii]] += 1
    
    print("    | Clust 1 | Clust 2 | Clust 3 | Clust 4 |")
    print("----|---------|---------|---------|---------|")
    for ii in range(4):
        prargs = tuple([ii+1] + list(classtable[ii,:]))
        print(" D{0} |      {1:>2} |      {2:>2} |      {3:>2} |      {4:>2} |".format(*prargs))
    
