#!/usr/bin/env python

'''
Experimental variogram script. Largely based on the Matlab script by Wolfgang Schwanghart
(http://www.mathworks.nl/matlabcentral/fileexchange/20355-experimental-semi-variogram).
'''

import numpy as np
import math
from scipy.spatial import distance
import itertools

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
        distance    = 
        gamma       = 
    
    '''
    
    # convert to numpy array, if necessary
    X = np.asanyarray(X)
    y = np.asanyarray(y)
    
    N = X.shape[0]
    dims = X.shape[1]
    
    assert len(y.shape) == 1, "y should be single column."
    assert N == len(y), "Number of coordinates and data values should be equal."
    if dims != 2 and theta is not None:
        print("Anisotropy analysis only on 2D data. Skipping.")
        theta = None
    assert 360 % theta == 0, "Please choose a number for theta that 360 is divisible with."
    
    #TODO: check for missings?
    
    maxDist = distance.euclidian((np.max(X, axis=1), np.min(X, axis=1))
    maxD = maxDist * maxDistFrac
    
    if subSample < 1:
        rndx = np.random.randint(0, N, round(subSample * N))
        X = X[rndx, :]
        y = y[rndx]
    
    # bin tolerance
    tol = maxDist / bins
    
    # coordinate combinations indices
    combs = np.array(itertools.combinations(range(N), 2))
    
    # calculate distances and merg with coordinate indices
    distVec = distance.pdist(X, metric='euclidian') ** 2
    # condensed distance matrix with coordinate indices
    condDist = np.hstack((combs, distVec))
    condDist = condDist[distVec > maxD]
    
    valDist = distance.euclidian(y[condDist[:,0], y[condDist[:,1])**2
    
    if thetaStep:
        noTheta = int(180 / theta);
        # convert to radians
        thetaStep = thetaStep / 180 * math.pi
        
        # calculate angles, clockwise from top
        theta = math.atan2( X[condDist[:,1],0] - X[condDist[:,0],0], \
                            X[condDist[:,1],1] - X[condDist[:,0],1] )
        
        # only semicircle
        theta[theta < 0] += math.pi
        theta[theta >= math.pi - thetaStep/2] = 0
        
        # bin the thetas, from 0 to 180 degrees
        nTheta, edgeTheta = np.histogram(theta, bins=noTheta,
                                        range=(-thetaStep/2, math.pi - thetaStep/2))
        # centers of the bins
        centTheta = edgeTheta[1:] + thetaStep/2
        centTheta[-1] = math.pi
    
    varFunc = lambda x: 1/(2 * len(x)) * sum(x)
    
    # bin the distances
    nDist, edgeDist = np.histogram(condDist[:,2], bins=bins, range=(0, maxDist))
    
    if thetaStep:
        
    return vals, centTheta, nums, distance

def plot_variogram():
    pass

def plot_variogram_cloud(binned=False):
    pass

if __name__ == '__main__':
    x = np.random.rand(1000,1)*4-2;  
    y = np.random.rand(1000,1)*4-2;
    z = 3*sin(x*15)+ np.random.randn(len(x));
    
    scatter(x,y,4,z,'filled'); box on;
    ylabel('y'); xlabel('x')
    title('data (coloring according to z-value)')
    
    hist(z,20)
    ylabel('frequency'); xlabel('z')
    title('histogram of z-values')
    
    d = variogram([x y],z,'plotit',true,'nrbins',50);
    title('Isotropic variogram')
    
    d2 = variogram([x y],z,'plotit',true,'nrbins',50,'anisotropy',true);
    title('Anisotropic variogram')

###################################################################################################
switch params.type
    case {'default','gamma'}
        % variogram anonymous function
        fvar     = @(x) 1./(2*numel(x)) * sum(x);
        % distance bins
        edges      = linspace(0,params.maxdist,params.nrbins+1);
        edges(end) = inf;

        [nedge,ixedge] = histc(iid(:,3),edges);
        
        if params.anisotropy
            S.val      = accumarray([ixedge ixtheta],lam,...
                                 [numel(edges) numel(thetaedges)],fvar,nan);
            S.val(:,end)=S.val(:,1); 
            S.theta    = thetacents;
            S.num      = accumarray([ixedge ixtheta],ones(size(lam)),...
                                 [numel(edges) numel(thetaedges)],@sum,nan);
            S.num(:,end)=S.num(:,1);                 
        else
            S.val      = accumarray(ixedge,lam,[numel(edges) 1],fvar,nan);     
            S.num      = accumarray(ixedge,ones(size(lam)),[numel(edges) 1],@sum,nan);
        end
        S.distance = (edges(1:end-1)+tol/2)';
        S.val(end,:) = [];
        S.num(end,:) = [];

    case 'cloud1'
        edges      = linspace(0,params.maxdist,params.nrbins+1);
        edges(end) = inf;
        
        [nedge,ixedge] = histc(iid(:,3),edges);
        
        S.distance = edges(ixedge) + tol/2;
        S.distance = S.distance(:);
        S.val      = lam;  
        if params.anisotropy            
            S.theta   = thetacents(ixtheta);
        end
    case 'cloud2'
        S.distance = iid(:,3);
        S.val      = lam;
        if params.anisotropy            
            S.theta   = thetacents(ixtheta);
        end
end


if params.plotit
    switch params.type
        case {'default','gamma'}
            marker = 'o--';
        otherwise
            marker = '.';
    end
    if ~params.anisotropy
        plot(S.distance,S.val,marker);
        axis([0 params.maxdist 0 max(S.val)*1.1]);
        xlabel('h');
        ylabel('\gamma (h)');
        title('(Semi-)Variogram');
    else
        [Xi,Yi] = pol2cart(repmat(S.theta,numel(S.distance),1),repmat(S.distance,1,numel(S.theta)));
        surf(Xi,Yi,S.val)
        xlabel('h y-direction')
        ylabel('h x-direction')
        zlabel('\gamma (h)')
        title('directional variogram')
    end
end
