#!/usr/bin/python

import os
import numpy as np

from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix, DefaultViewConverter
from pylearn2.datasets import preprocessing
import pylearn2.utils.serial as serial

DATA_DIR = 'home/nico/datasets/Kaggle/Digits/'

def initial_read():
    #create the training & test sets, skipping the header row with [1:]
    dataset = genfromtxt(open(DATA_DIR+'train.csv','r'), delimiter=',', dtype='f8')[1:]    
    targets = [x[0] for x in dataset]
    train = [x[1:] for x in dataset]
    test = genfromtxt(open(DATA_DIR+'test.csv','r'), delimiter=',', dtype='f8')[1:]
    
    # transform
    train = np.reshape([-1,28,28], np.array(train))
    test = np.reshape([-1,28,28], np.array(test))
    targets = np.array(targets)
    
    # pickle
    np.save(DATA_DIR+'train', train)
    np.save(DATA_DIR+'test', test)
    np.save(DATA_DIR+'targets', targets)
    
class Digits(DenseDesignMatrix):
    
    def __init__(self, which_set, start=None, stop=None, preprocessor=None):
        assert which_set in ['train','test']
        
        X = np.load(os.path.join(DATA_DIR,which_set+'.npy'))
        X = np.cast['float32'](X)
        # X needs to be 1D, shape info is stored in view_converter
        X = np.reshape(X,(X.shape[0], np.prod(X.shape[1:])))
        
        if which_set == 'test':
            # dummy targets
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
            
        # 2D data with 1 channel
        # do not change in case you extract patches, pylearn handles this!
        view_converter = DefaultViewConverter((28,28,1))
        
        super(Digits,self).__init__(X=X, y=y, view_converter=view_converter)
        
        assert not np.any(np.isnan(self.X))
        
        if preprocessor:
            preprocessor.apply(self)

def get_dataset(tot=False):
    if not os.path.exists(DATA_DIR+'train.npy') or
        not os.path.exists(DATA_DIR+'test.npy') or
        not os.path.exists(DATA_DIR+'targets.npy'):
        initial_read()
    
    train_path = DATA_DIR+'train_preprocessed.pkl'
    valid_path = DATA_DIR+'valid_preprocessed.pkl'
    tottrain_path = DATA_DIR+'tottrain_preprocessed.pkl'
    test_path = DATA_DIR+'test_preprocessed.pkl'
    
    if os.path.exists(train_path) and os.path.exists(valid_path) and os.path.exists(test_path):
        
        print 'loading preprocessed data'
        trainset = serial.load(train_path)
        validset = serial.load(valid_path)
        if tot:
            tottrainset = serial.load(tottrain_path)
        testset = serial.load(test_path)
    else:
        
        print 'loading raw data...'
        trainset = Digits(which_set='train', start=0, stop=40000)
        validset = Digits(which_set='train', start=40000, stop=47841)
        tottrainset = Digits(which_set='train')
        testset = Digits(which_set='test')
        
        print 'preprocessing data...'
        pipeline = preprocessing.Pipeline()
        
        #pipeline.items.append(preprocessing.ExtractGridPatches(patch_shape=(16,16),patch_stride=(8,8)))
        pipeline.items.append(preprocessing.GlobalContrastNormalization(sqrt_bias=10., use_std=True))
        # ZCA = zero-phase component analysis
        # very similar to PCA, but preserves the look of the original image better
        pipeline.items.append(preprocessing.ZCA())
        
        trainset.apply_preprocessor(preprocessor=pipeline, can_fit=True)
        # this uses numpy format for storage instead of pickle, for memory reasons
        trainset.use_design_loc(DATA_DIR+'train_design.npy')
        # note the can_fit=False: no sharing between train and valid data
        validset.apply_preprocessor(preprocessor=pipeline, can_fit=False)
        validset.use_design_loc(DATA_DIR+'valid_design.npy')
        tottrainset.apply_preprocessor(preprocessor=pipeline, can_fit=True)
        tottrainset.use_design_loc(DATA_DIR+'tottrain_design.npy')
        # note the can_fit=False: no sharing between train and test data
        testset.apply_preprocessor(preprocessor=pipeline, can_fit=False)
        testset.use_design_loc(DATA_DIR+'test_design.npy')
        
        # this path can be used for visualizing weights after training is done
        trainset.yaml_src = '!pkl: "%s"' % train_path
        validset.yaml_src = '!pkl: "%s"' % valid_path
        tottrainset.yaml_src = '!pkl: "%s"' % tottrain_path
        testset.yaml_src = '!pkl: "%s"' % test_path
        
        print 'saving preprocessed data...'
        serial.save(DATA_DIR+'train_preprocessed.pkl', trainset)
        serial.save(DATA_DIR+'valid_preprocessed.pkl', validset)
        serial.save(DATA_DIR+'tottrain_preprocessed.pkl', tottrainset)
        serial.save(DATA_DIR+'test_preprocessed.pkl', testset)
        
    if tot:
        return tottrainset, validset, testset
    else:
        return trainset, validset, testset
    
if __name__ == '__main__':
    pass
