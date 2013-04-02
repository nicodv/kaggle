#!/usr/bin/python

import os
import numpy as np

from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix, DefaultViewConverter
from pylearn2.datasets import preprocessing
import pylearn2.utils.serial as serial

DATA_DIR = '/home/nico/datasets/Kaggle/Whales/'


class Whales(DenseDesignMatrix):
    
    def __init__(self, which_set, which_data, start=None, stop=None, preprocessor=None):
        assert which_set in ['train','test']
        assert which_data in ['melspectrum','specfeat']
        
        X = np.load(os.path.join(DATA_DIR,which_set+which_data+'.npy'))
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
            
        if which_data == 'melspectrum':
            view_converter = DefaultViewConverter((67,40,1))
        elif which_data == 'specfeat':
            view_converter = DefaultViewConverter((67,1,24))
            
        super(Whales,self).__init__(X=X, y=y, view_converter=view_converter)
        
        assert not np.any(np.isnan(self.X))
        
        if preprocessor:
            preprocessor.apply(self)


def get_dataset(which_data, tot=False):
    train_path = DATA_DIR+'train'+which_data+'_preprocessed.pkl'
    valid_path = DATA_DIR+'valid'+which_data+'_preprocessed.pkl'
    tottrain_path = DATA_DIR+'tottrain'+which_data+'_preprocessed.pkl'
    test_path = DATA_DIR+'test'+which_data+'_preprocessed.pkl'
    
    if os.path.exists(train_path) and os.path.exists(valid_path) and os.path.exists(test_path):
        
        print 'loading preprocessed data'
        trainset = serial.load(train_path)
        validset = serial.load(valid_path)
        if tot:
            tottrainset = serial.load(tottrain_path)
        testset = serial.load(test_path)
    else:
        
        print 'loading raw data...'
        trainset = Whales(which_set="train", which_data=which_data, start=0, stop=56671)
        validset = Whales(which_set="train", which_data=which_data, start=56671, stop=66671)
        tottrainset = Whales(which_set="train", which_data=which_data)
        testset = Whales(which_set="test", which_data=which_data)
        
        print 'preprocessing data...'
        pipeline = preprocessing.Pipeline()
        
        if which_data == 'melspectrum':
            pipeline.items.append(preprocessing.Standardize(global_mean=True, global_std=True))
            # ZCA = zero-phase component analysis
            # very similar to PCA, but preserves the look of the original image better
            pipeline.items.append(preprocessing.ZCA())
        else:
            # global_mean/std=False voor per-feature standardization
            pipeline.items.append(preprocessing.Standardize(global_mean=False, global_std=False))
        
        trainset.apply_preprocessor(preprocessor=pipeline, can_fit=True)
        # this uses numpy format for storage instead of pickle, for memory reasons
        trainset.use_design_loc(DATA_DIR+'train_'+which_data+'_design.npy')
        validset.apply_preprocessor(preprocessor=pipeline, can_fit=False)
        validset.use_design_loc(DATA_DIR+'valid_'+which_data+'_design.npy')
        tottrainset.apply_preprocessor(preprocessor=pipeline, can_fit=True)
        tottrainset.use_design_loc(DATA_DIR+'tottrain_'+which_data+'_design.npy')
        # note the can_fit=False: no sharing between train and test data
        testset.apply_preprocessor(preprocessor=pipeline, can_fit=False)
        testset.use_design_loc(DATA_DIR+'test_'+which_data+'_design.npy')
        
        # this path can be used for visualizing weights after training is done
        trainset.yaml_src = '!pkl: "%s"' % train_path
        validset.yaml_src = '!pkl: "%s"' % valid_path
        tottrainset.yaml_src = '!pkl: "%s"' % tottrain_path
        testset.yaml_src = '!pkl: "%s"' % test_path
        
        print 'saving preprocessed data...'
        serial.save(DATA_DIR+'train'+which_data+'_preprocessed.pkl', trainset)
        serial.save(DATA_DIR+'valid'+which_data+'_preprocessed.pkl', validset)
        serial.save(DATA_DIR+'tottrain'+which_data+'_preprocessed.pkl', tottrainset)
        serial.save(DATA_DIR+'test'+which_data+'_preprocessed.pkl', testset)
        
    if tot:
        return tottrainset, None, testset
    else:
        return trainset, validset, testset

if __name__ == '__main__':
    pass
