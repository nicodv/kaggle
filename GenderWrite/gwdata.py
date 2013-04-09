#!/usr/bin/python

from pylearn2.datasets import preprocessing
from pylearn2.utils import serial
from collections import OrderedDict
import numpy as np
import pandas as pd
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix, DefaultViewConverter
from PIL import Image
from sklearn import decomposition

DATA_DIR = '/home/nico/datasets/Kaggle/GenderWrite/'

class GWData(DenseDesignMatrix):
    
    def __init__(self, which_set, start=None, stop=None, axes=('c', 0, 1, 'b')):
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
                    im = im.resize((im.size[0]//self.scale_factor,im.size[1]//self.scale_factor), Image.ANTIALIAS)
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
            y = np.save(DATA_DIR+'targets_per_page.npy')
        
        view = list(self.patch_size)
        view.extend([1])
        view_converter = DefaultViewConverter(view, axes)
        
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

def generate_patches():
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


def process_features():
    for curstr in ('train','test'):
        df = pd.read_csv(DATA_DIR+curstr+'.csv', delimiter=',')
        
        # convert to numeric
        df.language[df.language=='English'] = -1
        df.language[df.language=='Arabic'] = 1
        
        # delete unused columns
        df = df.drop(['same_text','writer'],axis=1)
        
        # fill missings with median per page
        df = df.groupby('page_id')
        f = lambda x: x.fillna(x.median())
        df = df.transform(f)
        df = df.reset_index()
        
        # remove features that have zero standard deviation
        df = df.iloc[:,df.std(axis=0) > 0]
        
        # do a PCA and keep largest components
        pca = decomposition.PCA(n_components=120, copy=False, whiten=True)
        #pca = decomposition.KernelPCA(n_components=120, kernel='linear')
        df = pca.fit_transform(np.array(df))
        
        np.save(DATA_DIR+'feat_'+curstr+'.npy', df)
    

def process_targets():
    targets = np.genfromtxt(DATA_DIR+'train_answers.csv', delimiter=',', filling_values=0, skip_header=1)[:,1]
    # make one-hot vector
    targets = np.array((targets, -targets+1)).T
    # targets per page, for combination model
    targets_pp = np.repeat(targets,[4]*targets.shape[0], axis=0)
    np.save(DATA_DIR+'targets_per_page.npy', targets_pp)
    np.save(DATA_DIR+'targets.npy', targets)
    

if __name__ == '__main__':
    generate_patches()
    process_features()
    process_targets()
