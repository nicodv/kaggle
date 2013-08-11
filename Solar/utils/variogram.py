#!/usr/bin/env python

'''
Experimental variogram script. Largely based on the Matlab script by Wolfgang Schwanghart
(http://www.mathworks.nl/matlabcentral/fileexchange/20355-experimental-semi-variogram).
'''

import numpy as np
import cmath
import math
from scipy.spatial import distance
import itertools
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


def variogram(X, y, bins=20, maxDistFrac=0.5, subSample=1., thetaStep=30):
    '''Calculates experimental variogram.
    
    Inputs:
        X           = array with coordinates [no. points * no. dimensions]
        y           = 1D array with values on locations in X
        bins        = number of bins that distance is grouped into
        maxDistFrac = fraction of maximum distance in dataset to consider as maximum distance
                      for variogram calculation
        subSample   = fraction of data that is used (in case of large data set)
        thetaStep   = step size for angle in anisotropy analysis, in degrees;
                      None means no anisotropy analysis
    Outputs:
        varData = dict with the following keys:
                    X           = input X
                    y           = input y
                    distance    = distances between points
                    bindistance = binned distances between points
                    gamma       = gamma values describing variogram
                    theta       = angles
                    bincount    = bincount of gammas
    
    '''
    
    # convert to numpy array, if necessary
    X = np.asanyarray(X)
    y = np.asanyarray(y)
    
    N = X.shape[0]
    dims = X.shape[1]
    
    assert y.shape[1] == 1 or len(y.shape) == 1, "y should be single column."
    assert N == len(y), "Number of coordinates and data values should be equal."
    if dims != 2 and thetaStep is not None:
        print("Anisotropy analysis only on 2D data. Skipping.")
        thetaStep = None
    assert thetaStep is None or 360 % thetaStep == 0, \
        "Please choose a number for theta that 360 is divisible with."
    
    #TODO: check for missings?
    
    maxDist = distance.euclidean(np.max(X, axis=0), np.min(X, axis=0))
    maxD = maxDist * maxDistFrac
    
    if subSample < 1:
        rndx = np.random.randint(0, N, round(subSample * N))
        X = X[rndx, :]
        y = y[rndx]
    
    # bin tolerance
    tol = maxDist / bins
    
    # coordinate indices combinations
    combs = np.array(zip(*[x for x in itertools.combinations(range(N), 2)]))
    
    # calculate condensed distance matrix between points
    XDist = distance.pdist(X, metric='euclidean')
    
    # calculate squared Euclidian distance between values
    yDist = np.array(map(distance.euclidean, y[combs[0,:]], y[combs[1,:]])) ** 2
    
    if thetaStep:
        nThetaSteps = int(180. / thetaStep);
        # convert to radians
        thetaStep = (thetaStep / 180.) * math.pi
        
        # calculate angles, clockwise from top
        theta = np.array([math.atan2(xx, yy) for xx, yy in \
                zip( X[combs[1,:],0] - X[combs[0,:],0], X[combs[1,:],1] - X[combs[0,:],1] )])
        
        # only semicircle
        theta[theta < 0] += math.pi
        theta[theta >= math.pi - thetaStep/2] = 0
        
        # bin the thetas, from 0 to 180 degrees
        thetaCount, thetaEdge = np.histogram(theta, bins=nThetaSteps, density=False,
                                        range=(-thetaStep/2, math.pi - thetaStep/2))
        # bin indices for all values in theta
        thetaInd = np.digitize(theta, thetaEdge[1:])
        
        # centers of the bins
        thetaCent = thetaEdge + thetaStep/2
        
        theta = thetaCent[thetaInd]
        thetabin = thetaCent[:-1]
    else:
        theta = None
        thetabin = None
    
    varFunc = lambda x: 1. / (2 * len(x)) * sum(x)
    
    # bin the distances and count, everything larger than maxD in a single bin
    distEdge = np.linspace(0, maxD, bins+1)
    distEdge[-1] = np.inf
    distCount,_ = np.histogram(XDist, bins=distEdge, density=False)
    distEdge = distEdge[1:]
    
    # bin indices for all distance values
    distInd = np.digitize(XDist, distEdge)
    
    if thetaStep:
        gamma = np.empty((len(distEdge), len(thetaEdge)-1))
        gamma[:] = np.nan
        nums = np.empty((len(distEdge), len(thetaEdge)-1))
        nums[:] = np.nan
        for d in np.unique(distInd):
            inds = np.where(distInd == d)
            sel = yDist[inds]
            for t in np.unique(thetaInd[inds]):
                tinds = np.where(thetaInd[inds] == t)
                tsel = sel[tinds]
                gamma[d,t] = np.nansum((gamma[d,t],varFunc(tsel)))
                nums[d,t] = np.nansum((nums[d,t], len(tsel)))
    else:
        gamma = np.empty(len(distEdge))
        gamma[:] = np.nan
        nums = np.empty(len(distEdge))
        nums[:] = np.nan
        for d in np.unique(distInd):
            sel = yDist[np.where(distInd == d)]
            gamma[d] = np.nansum((gamma[d],varFunc(sel)))
            nums[d] = np.nansum((nums[d], len(sel)))
    
    return {'X': X,
            'y': y,
            'distance': XDist,
            'bindistance': distEdge[distInd] + tol/2,
            'maxD': maxD,
            'distbin': distEdge[:-1] + tol/2,
            'ydistance': yDist,
            'gamma': gamma[:-1],
            'theta': theta,
            'thetabin': thetabin,
            'bincount': nums[:-1]
            }

def plot_variogram(ax, dist, gamma, maxD=None, theta=None, cloud=False):
    marker = 'k.' if cloud else 'ro--'
    
    if isinstance(theta, np.ndarray):
        Ci = [cmath.rect(x, y) for x, y in itertools.product(dist, theta)]
        Ci = zip(*[(x.real, x.imag) for x in Ci])
        Xc, Yc = np.meshgrid(Ci[0], Ci[1])
        ax.plot_surface(Xc.flatten(), Yc.flatten(), gamma, cmap=cm.jet, antialiased=False)
        ax.set_xlabel(r"Distance $h_x$")
        ax.set_ylabel(r"Distance $h_y$")
        ax.set_zlabel(r"$\gamma (h)$")
    else:
        ax.plot(dist, gamma, marker)
        ax.set_xlim((0,maxD))
        ax.set_ylim((0,1.1 * np.nanmax(gamma)))
        ax.set_xlabel("Distance h")
        ax.set_ylabel(r"$\gamma (h)$")
    return

if __name__ == '__main__':
    x = np.random.rand(100,1)*4 - 2
    y = np.random.rand(100,1)*4 - 2
    z = 3*np.sin(x*15) + np.random.randn(len(x),1)
    varData = variogram(np.hstack((x, y)), z, bins=50, maxDistFrac=0.5, subSample=1., thetaStep=30)
    dist, bdist, distbin, ydist, gamma, maxD, theta, thetabin = varData['distance'], \
            varData['bindistance'], varData['distbin'], varData['ydistance'], \
            varData['gamma'], varData['maxD'], varData['theta'], varData['thetabin']
    
    fig = plt.figure(1)
    ax = fig.add_subplot(2, 3, 1)
    ax.scatter(x, y, marker='o', c=z)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Data (coloring according to z-value)")
    
    ax = fig.add_subplot(2, 3, 2)
    ax.hist(z, 20)
    ax.set_xlabel("z")
    ax.set_ylabel("frequency")
    ax.set_title("Histogram of z-values")
    
    ax = fig.add_subplot(2, 3, 3)
    plot_variogram(ax, bdist, ydist, maxD, cloud=True)
    ax.set_title("Variogram cloud (binned distances)")
    ax = fig.add_subplot(2, 3, 4)
    plot_variogram(ax, dist, ydist, maxD, cloud=True)
    ax.set_title("Variogram cloud (raw distances)")
    
    ax = fig.add_subplot(2, 3, 5)
    plot_variogram(ax, distbin, gamma, maxD)
    ax.set_title("Isotropic variogram")
    ax.grid()
    
    ax = fig.add_subplot(2, 3, 6, projection='3d')
    plot_variogram(ax, distbin, gamma, maxD, thetabin)
    ax.set_title("Anisotropic variogram")
    
    fig = plt.figure(2)
    for ii, th in enumerate(thetabin):
        ax = fig.add_subplot(2, 3, ii+1)
        plot_variogram(ax, distbin, gamma[:,ii], maxD, th)
        ax.set_title("Variogram for theta = %d degrees" % ((th/math.pi)*180, ))
    
    plt.show()
