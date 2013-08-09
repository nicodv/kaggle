import os
import json
import pandas as pd
import numpy as np
import netCDF4

with open('../SETTINGS.json') as data_file:
    basePath = json.load(data_file)['base_path']


def load_stations(fileName='station_info.csv'):
    path = os.path.join(basePath, fileName)
    stations = pd.read_csv(path, skiprows=1, index_col=0, dtype='np.float64')
    stations.columns = ['lat', 'lon', 'elev']
    return stations

def load_mesonet(fileName='train.csv'):
    path = os.path.join(basePath, fileName)
    mesoData = pd.read_csv(path, index_col=0, parse_dates=True, dtype='np.float64')
    return mesoData

def load_gefs(dataset):
    if dataset == 'train':
        fileName = 'gefs_train.nc'
    elif dataset == 'test':
        fileName = 'gefs_test.nc'
    else:
        raise Exception("Unknown data set.")
    path = os.path.join(basePath, fileName)
    gefsData = netCDF4.Dataset(path)
    return gefsData
