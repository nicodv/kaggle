#!/usr/bin/python

import numpy as np
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix, DefaultViewConverter
from PIL import Image

DATA_DIR = '/home/nico/datasets/Kaggle/GenderWrite/'

class GWData(DenseDesignMatrix):
    
    def __init__(self, which_set, start=None, stop=None):
        assert which_set in ['train','test']
        
        # size of patches extracted from JPG images
        self.patch_size = (30,30)
        # how many patches will be drawn for each of the writers
        self.no_patches = 600
        # downsample data?
        self.scale_factor = 2
        # threshold for standard deviation above which there seems to be a signal
        self.stdthreshold = 5
        # extra whitespace (if any) around writing that will be included
        self.wspace = 16
        
        if start is None:
            if which_set == 'train':
                writers = range(1,283)
            else:
                writers = range(283,476)
        else:
            assert start >= 1
            assert start < stop
            assert stop <= 476
            writers = range(start,stop)
        
        X = np.zeros((len(writers)*self.no_patches, self.patch_size[0], self.patch_size[1], 1))
        for ii, writer in enumerate(writers):
            # files are named 0001_1.jpg, etc.
            wrstr = str(writer).zfill(4)
            images = []
            for page in range(1,5):
                # open and convert to grayscale
                im = Image.open(DATA_DIR+'jpgs/'+wrstr+'_'+str(page)+'.jpg').convert('L')
                # resize
                if self.scale_factor not in (None, 1, 1.):
                    im.resize((im.size[0]//self.scale_factor,im.size[1]//self.scale_factor), Image.ANTIALIAS)
                # from here on, work with 2D numpy array
                im = np.squeeze(np.array(im, dtype=np.uint8))
                # crop
                im = self.__crop_image(im)
                images.append(im)
            
            # extract patches
            X[ii*self.no_patches:(ii+1)*self.no_patches,:,:,0] = self.__extract_patches(images)
        
        X = np.reshape(X,(X.shape[0],np.prod(X.shape[1:])))
        
        if which_set == 'test':
            y = np.zeros((X.shape[0],2))
        else:
            y = np.genfromtxt(DATA_DIR+'train_answers.csv', delimiter=',', filling_values=0, skip_header=1)
        # shape y like X, so repeat y[x] no_patches times
        y = np.repeat(y, self.no_patches)
        # make one-hot vector
        y = np.array((y, -y+1)).T
        
        view = list(self.patch_size)
        view.extend([1])
        view_converter = DefaultViewConverter(view)
        
        super(GWData,self).__init__(X=X, y=y, view_converter=view_converter)
        
        assert not np.any(np.isnan(self.X))
        
    def __crop_image(self, im):
        '''Crop a numpy array that represents a grayscale image.
        '''
        
        colstd = np.std(im, axis=1)
        rowstd = np.std(im, axis=0)
        
        # now find signals above threshold, from 4 directions
        crops = []
        for stdvec in (colstd, colstd[::-1], rowstd, rowstd[::-1]):
            for ii, cur in enumerate(stdvec):
                if cur > self.stdthreshold:
                    crops.append(np.max((0,ii-self.wspace)))
                    break
        # now crop the image
        im = im[crops[0]:-(crops[1]+1), crops[2]:-(crops[3]+1)]
        print 'cropped with: %s, %s, %s, %s' % tuple(crops)
        return im
    
    def __extract_patches(self, images):
        '''Takes a list of 4 images (which are 2D numpy arrays) and returns patches.
        '''
        assert self.no_patches % 4 == 0
        no_patches_per = self.no_patches // 4
        
        patches = np.zeros((self.no_patches,self.patch_size[0],self.patch_size[1]))
        for jj, image in enumerate(images):
            for ii in range(no_patches_per):
                while True:
                    row = np.random.randint(0,image.shape[0]-self.patch_size[0])
                    col = np.random.randint(0,image.shape[1]-self.patch_size[1])
                    patch = image[row:row+self.patch_size[0],col:col+self.patch_size[1]]
                    if np.std(patch) > self.stdthreshold:
                        break
                
                patches[jj*no_patches_per+ii,:,:] = patch
        
        return patches

if __name__ == '__main__':
    pass
