#!/usr/bin/python2

from pylearn2.datasets import preprocessing
from pylearn2.utils import serial
from collections import OrderedDict
import numpy as np
import pandas as pd
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix, DefaultViewConverter
from PIL import Image
from sklearn import decomposition

import GalaxyZoo.gzdata


DATA_DIR = '/home/nico/Data/GalaxyZoo/'


#!/usr/bin/python

'''
Data class in pylearn2 format
'''

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

        if which_data == 'melspectrum':
            # 2D data with 1 channel
            view_converter = DefaultViewConverter((67,40,1))
        elif which_data == 'specfeat':
            # 24 channels with 1D data
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
        # note the can_fit=False: no sharing between train and test data
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
        return tottrainset, validset, testset
    else:
        return trainset, validset, testset

if __name__ == '__main__':
    pass




class GZData(DenseDesignMatrix):

    def __init__(self, which_set, start=None, stop=None, axes=('c', 0, 1, 'b')):
        assert which_set in ['train', 'test']

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
            y = np.load(DATA_DIR+'targets_per_page.npy')

        view = list(self.patch_size)
        view.extend([1])
        view_converter = DefaultViewConverter(view, axes)

        super(GZData,self).__init__(X=X, y=y, view_converter=view_converter)

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
    datasets['train'] = GalaxyZoo.gzdata.GZData(which_set = 'train', start=1, stop=201)
    datasets['valid'] = GalaxyZoo.gzdata.GZData(which_set = 'train', start=201, stop=283)
    datasets['test'] = GalaxyZoo.gzdata.GZData(which_set = 'test')
    datasets['tottrain'] = GalaxyZoo.gzdata.GZData(which_set = 'train')

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

        can_fit = curstr == 'train'

        df = pd.read_csv(DATA_DIR+curstr+'.csv', delimiter=',')

        # delete unused columns
        writers = df['writer'] # save this one for later
        langs = df['language']
        df = df.drop(['writer','language','same_text'],axis=1)

        def winsorize(col, stat='std', factor=3):
            cutoff = eval('factor * np.' + stat + '(col)')
            colmean = np.mean(col)
            col[np.abs(col - colmean) > cutoff] = colmean + np.sign(col - colmean)*cutoff
            return col

        for colname in df.columns:
            gbcol = df.groupby(['page_id'])[colname]
            f = eval('lambda x: x.fillna(x.median())')
            col = gbcol.transform(f)

            df[colname] = winsorize(df[colname])

        df = df.drop(['page_id'],axis=1)

        # remove features that have zero standard deviation
        if can_fit:
            delmask = df.std(axis=0) > 0

        df = df.iloc[:, delmask]

        # combine features that have a lot of a single value into a single
        # (hopefully useful) feature by standardizing them and then taking the mean of all
        # (take into account when they have only a few unique values, might be useful still)
        if can_fit:
            sfcols = []
            for col in df.columns:
                if ((df[col] == df[col].median()).sum() > 0.6*len(df[col]) and np.unique(df[col]).count() >= 8) or \
                    ((df[col] == df[col].median()).sum() > 0.75*len(df[col]) and np.unique(df[col]).count() >= 4) or \
                    (df[col] == df[col].median()).sum() > 0.9*len(df[col]):
                    sfcols.append(col)

        sft = []
        for col in sfcols:
            mask = df[col]==df[col].median()
            colmean = np.mean(df[col][~mask])
            colstd = np.std(df[col][~mask])
            # set the median value to the mean of non-median values
            sf = df[col]
            # in test data a col might still have mean of nan, std of nan/0
            if not np.isnan(colstd) and colstd> 0:
                sf[mask] = colmean
                sf = (sf - colmean) / colstd
                sft.append(sf)

        # now add this aggregate feature to the feature dataframe
        df['sft'] = np.mean(np.array(sft).T, axis=1)
        # ... and drop the previous individual features
        df = df.drop(sfcols, axis=1)

        # standardize the data
        if can_fit:
            stdmean = df.mean(axis=0)
            stdstd = df.std(axis=0)

        df = (df - stdmean) / stdstd

        df = np.array(df)
        if can_fit:
            # do a PCA and keep all components
            decomp1 = decomposition.PCA(n_components=150, whiten=True)
            decomp2 = decomposition.FastICA(n_components=150, algorithm='deflation', whiten=True)
            decomp3 = decomposition.KernelPCA(n_components=150, kernel='rbf', degree=3)
            # only fit on train data
            decomp1 = decomp1.fit(df)
            decomp2 = decomp2.fit(df)
            decomp3 = decomp3.fit(df)

        df = np.concatenate((decomp1.transform(df), decomp2.transform(df), decomp3.transform(df)), axis=1)

        np.save(DATA_DIR+'feat_'+curstr+'.npy', df)

        print "finished with %s" % curstr


def process_targets():
    targets = np.genfromtxt(DATA_DIR+'training_solutions_rev1.csv', delimiter=',', filling_values=0, skip_header=1)[:,1]
    np.save(DATA_DIR+'targets.npy', targets)


if __name__ == '__main__':
    process_targets()
    #generate_patches()
    process_features()
