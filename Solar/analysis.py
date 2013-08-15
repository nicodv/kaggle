#/usr/bin/env python

from Solar.utils import datamanip, variogram
import matplotlib.pyplot as plt

# ANALYZE ANISOTROPY GEFS DATA
stations = datamanip.load_stations()
gefsData = datamanip.load_gefs('train')

# Let's work with the mean over the ensemble and over the forecasts
gefsData = np.apply_over_axes(np.mean, gefsData.variables.values()[-1], [1, 2])[0]
# let's work with the first ensemble member only, for the 9-hour forecast
#gefsData = gefsData.variables.values()[-1][:,0,2][0]


fig = plt.figure(1)
for ii, var in enumerate(variables):
    lats = gefsData.variables['lat'][:]
    lons = gefsData.variables['lon'][:]
    varData = variogram.variogram((lats, lons), variable, thetaStep=15)
    ax = fig.add_subplot(3, 5, ii+1, projection='3d')
    plot_variogram(ax, varData['distbin'], varData['gamma'], varData['maxD'], varData['thetabin'])
    ax.set_title("Anisotropic variogram for variable: {}".format(var))

plt.show()

# STATIONARITY MESONET DATA
#from statsmodels.tsa.stattools import adfuller
mesoData = datamanip.load_mesonet()
plt.figure(); mesoData.plot(); plt.legend(loc='best')


# SPATIAL DISPERSION
import rpy2.robjects as robjects
import itertools as iter
from scipy.spatial.distance import euclidean

stations = stations.T
for curVar in ('temp', ):
    curData = 
    # spatial dispersion is defined as the variance of the difference in time series
    spatialDisp = [np.var(curData[x] - curData[y]) for a, b in iter.combinations(stations.columns, 2)]
    distances = [euclidean(stations[a], stations[b]) for a, b in iter.combinations(stations.columns, 2)]


robjects.r('command')
