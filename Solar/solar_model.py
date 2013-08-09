import json
import numpy as np

def getGrid(data,date,fHour,eMember):
    """
	getGrid()
	Description: Load GEFS data from a specified date, forecast hour, and ensemble member.
	Parameters:
	data (Dataset) - Dataset object from loadData
	date (int) - date of model run in YYYYMMDD format
	fHour (int) - forecast hour
	eMember (int) - ensemble member id.
	Returns: numpy 2d array from the specified output
	"""
    dateIdx = np.where(data.variables['intTime'][:] == date)[0]
    fIdx = np.where(data.variables['fhour'][:] == fHour)[0]
    eIdx = np.where(data.variables['ens'][:] == eMember)[0]
    return data.variables.values()[-1][dateIdx,eIdx,fIdx][0]

def getDailyMeanSumGrid(data,date):
    """
	getDailyMeanSumGrid()
	Description: For a particular date, sums over all forecast hours for each ensemble member then takes the
	mean of the summed data and scales it by the GEFS time step.
	Parameters:
	data (Dataset) - netCDF4 object from loadData
	date (int) - date of model run in YYYYMMDD format
	Returns - numpy 2d array from the specified output
	"""
    dateIdx = np.where(data.variables['intTime'][:] == date)[0]
    fIdx = np.where(data.variables['fhour'][:] <= 24)[0]
    return data.variables.values()[-1][dateIdx,:,fIdx,:,:].sum(axis=2).mean(axis=1)[0] * 3600 * 3

def main():
    MesonetData,dates,stationdata = loadMesonetData(basePath + 'submission.csv')
    data = loadData(dataPath + "dswrf_sfc_latlon_subset_20080101_20121130.nc")
    lons = data.variables['lon'][:]
    lats = data.variables['lat'][:]
    f = open('spline_submission.csv', 'w')
    header = ["Date"]
    header.extend(stationdata['stid'].tolist())
    f.write(",".join(header) + "\n")
    for date in dates:
        print date
        grid = getDailyMeanSumGrid(data,date*100)
        outdata=buildSplines(data,grid,stationdata)
        outdata = outdata.reshape(outdata.shape[0],1)
        f.write("%d" % date + ",")
        np.savetxt(f, outdata.T, delimiter=',',fmt='%7.0f')
    f.close()
    data.close()

if __name__ == "__main__":
    main()
