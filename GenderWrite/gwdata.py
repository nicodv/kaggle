#!/usr/bin/python

import numpy as np
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix, DefaultViewConverter
from pylearn2.datasets import preprocessing
from pylearn2.utils import serial
from PIL import Image

DATA_DIR = '/home/nico/datasets/Kaggle/GenderWrite/'

class GWData(DenseDesignMatrix):
    
    def __init__(self, which_set, start=None, stop=None):
        assert which_set in ['train','test']
        
        # size of patches extracted from JPG images
        self.patch_size = (32,32)
        # how many patches will be drawn for each of the writers
        self.no_patches = 100
        # downsample data?
        self.scale_factor = 2
        # threshold for standard deviation above which there seems to be a signal
        self.stdthreshold = 1
        # extra whitespace (if any) around writing that will be included
        self.wspace = 16
        
        if start is None:
            if which_set == 'train':
                writers = range(1,283)
            else:
                writers = range(283,476)
        else:
            assert start >= 0
            assert start < stop
            assert stop < 476
            writers = range(start,stop)
        
        X = np.zeros(len(writers)*self.no_patches, self.patch_size[0], self.patch_size[1], 1)
        for writer in writers:
            # files are named 0001_1.jpg, etc.
            wrstr = str(writer).zfill(4)
            images = []
            for page in range(1,5):
                # open and convert to grayscale
                im = Image.open(DATA_DIR+wrstr+'_'+str(page)+'.jpg').convert('L')
                # resize
                if scale_factor not in (None, 1, 1.):
                    im.resize((im.size[0]/self.scale_factor,im.size[1]/self.scale_factor), Image.ANTIALIAS)
                # from here on, work with 2D numpy array
                im = np.squeeze(np.array(im, dtype=np.uint8))
                # crop
                im = __crop_image(im)
                images.append(im)
            
            # extract patches
            X[(writer-1)*no_patches:writer*self.no_patches,:,:,1] = __extract_patches(images)
        
        if which_set == 'test':
            y = np.zeros((X.shape[0],2))
        else:
            y = np.genfromtxt(DATA_DIR+'train_answers.csv', delimiter=',', filling_values=0, skip_header=1)
        # shape y like X, so repeat y[x] no_patches times
        y = np.repeat(y, self.no_patches)
        # make one-hot vector
        y = np.array((y, -y+1)).T
        
        view_converter = DefaultViewConverter(patch_size.extend(1))
        
        super(GWData,self).__init__(X=X, y=y, view_converter=view_converter)
        
        assert not np.any(np.isnan(self.X))
        
    def __crop_image(self, im):
        '''Crop a numpy array that represents a grayscale image.
        '''
        
        colstd = np.std(im, axis=1)
        rowstd = np.std(im, axis=0)
        
        # now find signals above threshold, from 4 directions
        crops = []
        for stdvec in (colstd, np.fliplr(colstd), rowstd, np.fliplr(rowstd)):
            for ii, cur in enumerate(stdvec):
                if cur > self.stdthreshold:
                    crop.append(np.max(0,ii-self.wspace))
                    break
        # now crop the image
        im = im[crop[0]:-(crop[1]+1), crop[2]:-(crop[3]+1)]
        return im
    
    def __extract_patches(self, images):
        '''Takes a list of 4 images (which are 2D numpy arrays) and returns patches.
        '''
        assert self.no_patches % 4 == 0
        no_patches_per = self.no_patches / 4
        
        patches = np.zeros((self.no_patches,self.patch_size[0],self.patch_size[1]))
        for jj, image in enumerate(images):
            for ii in range(no_patches_per):
                while True:
                    row = np.random.randint(0,image.shape[0]-self.patch_size[0])
                    col = np.random.randint(0,image.shape[1]-self.patch_size[1])
                    patch = image[row:row+self.patch_size[0],col:col+self.patch_size[1]]
                    if np.std(patch) > self.stdthreshold:
                        break
                
                patches[jj*self.no_patches_per+ii,:,:] = patch
        
        return patches

def gendata():
    datasets = {'train': GWData(which_set = 'train', start=0, stop=282),
                'valid': GWData(which_set = 'train', start=282, stop=475),
                'test': GWData(which_set = 'test', start=0, stop=282),
                'tottrain': GWData(which_set = 'train')}
    
    for dstr, dset in datasets.iteritems():
        # preprocess patches
        pipeline = preprocessing.Pipeline()
        pipeline.items.append(preprocessing.GlobalContrastNormalization())
        pipeline.items.append(preprocessing.ZCA())
        # only fit on train data
        trainbool = dstr == 'train' or dstr == 'tottrain'
        dset.apply_preprocessor(preprocessor=pipeline, can_fit=trainbool)
        # save
        dset.use_design_loc(dstr+'_design.npy')
        serial.save(DATA_DIR+'gw_preprocessed_'+dstr+'.pkl', dset)

if __name__ == '__main__':
    gendata()
