#!/usr/bin/python

from pylearn2.datasets import preprocessing
from pylearn2.utils import serial
from collections import OrderedDict
from GenderWrite.gwdata import GWData

DATA_DIR = '/home/nico/datasets/Kaggle/GenderWrite/'

def gendata():

    datasets = OrderedDict()
    datasets['train'] = GWData(which_set = 'train', start=1, stop=201)
    datasets['valid'] = GWData(which_set = 'train', start=201, stop=283)
    datasets['test'] = GWData(which_set = 'test')
    datasets['tottrain'] = GWData(which_set = 'train')
    
    # preprocess patches
    pipeline = preprocessing.Pipeline()
    pipeline.items.append(preprocessing.GlobalContrastNormalization())
    pipeline.items.append(preprocessing.ZCA())
    for dstr, dset in datasets.iteritems():
        print dstr
        # only fit on train data
        trainbool = dstr == 'train' or dstr == 'tottrain'
        dset.apply_preprocessor(preprocessor=pipeline, can_fit=trainbool)
        # save
        dset.use_design_loc(DATA_DIR+dstr+'_design.npy')
        serial.save(DATA_DIR+'gw_preprocessed_'+dstr+'.pkl', dset)

if __name__ == '__main__':
    gendata()