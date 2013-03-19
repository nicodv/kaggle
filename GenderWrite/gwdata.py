#!/usr/bin/python

import os
import numpy as np
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix, DefaultViewConverter
from pylearn2.datasets import preprocessing

DATA_DIR = '/home/nico/Code/datasets/Kaggle/GenderWrite/'

class GWData(DenseDesignMatrix):
    
    def __init__(self, which_set, start=None, stop=None, preprocessor=None):
        assert which_set in ['train','valid','test']
        
        t_set = 'train' if which_set=='valid' else which_set
        X = np.load(os.path.join(DATA_DIR,t_set+'.npy'))
        X = np.cast['float32'](X)
        X = np.reshape(X,(X.shape[0], np.prod(X.shape[1:])))
        
        if which_set == 'test':
            y = np.zeros((X.shape[0],2))
        else:
            y = np.load(os.path.join(DATA_DIR,'targets.npy'))
            
        if start is not None:
            assert start >= 0
            assert stop > start
            assert stop <= X.shape[0]
            X = X[start:stop, :]
            y = y[start:stop]
            assert X.shape[0] == y.shape[0]
            
        view_converter = DefaultViewConverter((36,36,1))
        
        super(GWData,self).__init__(X=X, y=y, view_converter=view_converter)
        
        assert not np.any(np.isnan(self.X))
        
        if preprocessor:
            preprocessor.apply(self)
