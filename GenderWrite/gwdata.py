#!/usr/bin/python

import os
import numpy as np
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix, DefaultViewConverter
from pylearn2.utils import serial
import Image

DATA_DIR = '/home/nico/Code/datasets/Kaggle/GenderWrite/'

class GWData(DenseDesignMatrix):
    
    def __init__(self, which_set, start=None, stop=None):
        assert which_set in ['train','test']
        
        if which_set == 'train':
            writers = range(1,283)
        else:
            writers = range(283,476)
            
        X = []
        for writer in writers:
            curstr = '0'+str(num)
            images = []
            for page in range(1,5):
                im = Image.open(DATA_DIR+writer+'_'+page+'.jpg'))
                # crop and resize
                im = im.crop((100,100,im.size[0]-100,im.size[1]-100))
                rfactor = 2
                im.resize((im.size[0]/rfactor,im.size[1]/rfactor, Image.ANTIALIAS)
                images.append(numpy.array(im.getdata()))
            # merge pages
            X.append(images)
        
        X = np.cast['float32'](X)
        imagesize = X.shape[1:]
        X = np.reshape(X,(X.shape[0], np.prod(imagesize)))
        
        if which_set == 'test':
            y = np.zeros((X.shape[0],2))
        else:
            y = np.genfromtxt(DATA_DIR+'train_answers.csv', delimiter=',', filling_values=0, skip_header=1)
            
        if start is not None:
            assert start >= 0
            assert stop > start
            assert stop <= X.shape[0]
            X = X[start:stop, :]
            y = y[start:stop]
            assert X.shape[0] == y.shape[0]
        
        view_converter = DefaultViewConverter((imagesize.extend(1)))
        
        super(GWData,self).__init__(X=X, y=y, view_converter=view_converter)
        
        assert not np.any(np.isnan(self.X))
        
def gendata():
    patch_shape = (32, 32)
    num_patches = [1e6, 5e5, 1e6, 1e6]
    
    datasets = [GWData(which_set = 'train', start=0, stop=283),
                GWData(which_set = 'train', start=283, stop=475),
                GWData(which_set = 'test', start=0, stop=283),
                GWData(which_set = 'train')]
    
    for ii, curstr in enumerate(('train', 'valid', 'test', 'tottrain')):
        # preprocess
        pipeline = preprocessing.Pipeline()
        pipeline.items.append(preprocessing.ExtractPatches(patch_shape=patch_shape, num_patches=num_patches[ii]))
        pipeline.items.append(preprocessing.GlobalContrastNormalization())
        pipeline.items.append(preprocessing.ZCA())
        trainbool = curstr == 'train' or curstr == 'tottrain'
        datasets[ii].apply_preprocessor(preprocessor=pipeline, can_fit=trainbool)
        # save
        use_design_loc(curstr+'_design.npy')
        serial.save('/home/nico/datasets/Kaggle/GenderWriting/gw_preprocessed_'+curstr+'.pkl', datasets[ii])

if __name__ == '__main__':
    gendata()
