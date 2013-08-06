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
                    gamma       = 
                    theta       = angles
                    bincount    = bincount of gammas
    
    '''
    
    # convert to numpy array, if necessary
    X = np.asanyarray(X)
    y = np.asanyarray(y)
    
    N = X.shape[0]
    dims = X.shape[1]
    
    assert len(y.shape) == 1, "y should be single column."
    assert N == len(y), "Number of coordinates and data values should be equal."
    if dims != 2 and thetaStep is not None:
        print("Anisotropy analysis only on 2D data. Skipping.")
        thetaStep = None
    assert 360 % thetaStep == 0, "Please choose a number for theta that 360 is divisible with."
    
    #TODO: check for missings?
    
    maxDist = distance.euclidian((np.max(X, axis=1), np.min(X, axis=1))
    maxD = maxDist * maxDistFrac
    
    if subSample < 1:
        rndx = np.random.randint(0, N, round(subSample * N))
        X = X[rndx, :]
        y = y[rndx]
    
    # bin tolerance
    tol = maxDist / bins
    
    # coordinate indices combinations
    combs = np.array(itertools.combinations(range(N), 2))
    
    # calculate condensed distance matrix between points
    XDist = distance.pdist(X, metric='euclidian') ** 2
    
    # calculate squared Euclidian distance between values
    yDist = distance.euclidian(y[XDist[:,0], y[XDist[:,1])**2
    
    if thetaStep:
        nThetaSteps = int(180 / thetaStep);
        # convert to radians
        thetaStep = thetaStep / 180 * math.pi
        
        # calculate angles, clockwise from top
        theta = math.atan2( X[combs[:,1],0] - X[combs[:,0],0], \
                            X[combs[:,1],1] - X[combs[:,0],1] )
        
        # only semicircle
        theta[theta < 0] += math.pi
        theta[theta >= math.pi - thetaStep/2] = 0
        
        # bin the thetas, from 0 to 180 degrees
        thetaCount, thetaEdge = np.histogram(theta, bins=nThetaSteps, density=False,
                                        range=(-thetaStep/2, math.pi - thetaStep/2))
        # bin indices for all values in theta
        thetaInd = np.digitize(theta, thetaEdge)
        
        # centers of the bins
        thetaCent = thetaEdge[1:] + thetaStep/2
        thetaCent[-1] = math.pi
    
    varFunc = lambda x: 1/(2 * len(x)) * sum(x)
    
    # bin the distances and count, everything larger than maxD in a single bin
    distEdge = np.linspace(0, maxD, bins+1)
    distEdge[-1] = np.inf
    distCount,_ = np.histogram(XDist, bins=distEdge, density=False)
    # bin indices for all distance values
    distInd = np.digitize(XDist, distEdge)
    
    if thetaStep:
        inds = np.hstack((distInd, thetaInd))
    else:
        inds = distInd
    
    indices = np.where(np.ediff1d(distInd, to_begin=[1], to_end=[1]))[0]
    vals_len = len(indices) - 1
    vals = np.zeros(vals_len)
    a_f = a.flat
    for i in xrange(vals_len):
        gamma[i] = varFunc(a_f[indices[i]:indices[i + 1]])
        nums[i] = sum(a_f[indices[i]:indices[i + 1]])
    
    return varData{'X': X,
                   'y': y,
                   'distance': XDist,
                   'bindistance': distEdge[distInd] + tol/2,
                   'gamma': gamma,
                   'theta': thetaCent[thetaInd],
                   'bincount': nums
                  }

def plot_variogram(ax, varData, anisotropy=False, cloud=False, binned=False):
    if cloud:
        marker = 'k.'
        distVar = varData['distance'] if not binned else varData['bindistance']
    else:
        marker = 'ro--'
        distVar = varData['distance']
    
    if anisotropy:
        Ci = zip(*[(x.real, x.imag) for x in itertools.imap(rect, distVar, varData['theta'])])
        Xc, Yc = np.meshgrid(Ci[0], Ci[1])
        surf = ax.plot_surface(Xc, Yc, varData['z'], rstride=1, cstride=1, cmap=cm.jet,
                               linewidth=0, antialiased=True, projection='3D')
        ax.set_xlabel('h y-direction')
        ax.set_ylabel('h x-direction')
        ax.set_zlabel(r"$ \gamma (h) $")
        ax.set_title("Directional variogram")
    else:
        ax.plot(distVar, varData['gamma'], marker)
        ax.xlim((0,varData['maxD']))
        ax.ylim((0,1.1 * max(varData['gamma'])))
        ax.set_xlabel("h");
        ax.set_ylabel(r"$ \gamma (h) $");
        ax.set_title("(Semi-)Variogram");
    return

if __name__ == '__main__':
    x = np.random.rand(1000,1)*4 - 2  
    y = np.random.rand(1000,1)*4 - 2
    z = 3*sin(x*15)+ np.random.randn(len(x))
    varData, distData,  = variogram(np.hstack((x, y)), z, bins=50, maxDistFrac=0.5, subSample=1., thetaStep=30)
    
    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex=False, sharey=False)
    
    ax1.scatter(x, y, marker='o', c=z)
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_title("Data (coloring according to z-value)")
    
    ax2.hist(z, 20)
    ax2.set_xlabel("z")
    ax2.set_ylabel("frequency")
    ax2.set_title("Histogram of z-values")
    
    plot_variogram(ax3, varData);
    ax3.set_title("Isotropic variogram")
    
    plot_variogram(ax4, varData, anitropy=True)
    ax4.set_title("Anisotropic variogram")
    
    plt.show()
