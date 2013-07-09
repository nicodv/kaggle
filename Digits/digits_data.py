#!/usr/bin/python

import os
import numpy as np

from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from pylearn2.datasets import preprocessing
import pylearn2.utils.serial as serial

from PIL import Image

rng = np.random.RandomState(42)

DATA_DIR = '/home/nico/datasets/Kaggle/Digits/'

def initial_read():
    #create the training & test sets, skipping the header row with [1:]
    dataset = np.genfromtxt(open(DATA_DIR+'train.csv','r'), delimiter=',', dtype='f8')[1:]
    targets = [x[0] for x in dataset]
    train = [x[1:] for x in dataset]
    test = np.genfromtxt(open(DATA_DIR+'test.csv','r'), delimiter=',', dtype='f8')[1:]
    
    # transform
    train = np.reshape(np.array(train), (-1,28,28))
    test = np.reshape(np.array(test), (-1,28,28))
    targets = np.array(targets)
    
    # pickle
    np.save(DATA_DIR+'train', train)
    np.save(DATA_DIR+'test', test)
    np.save(DATA_DIR+'targets', targets)
    
class Digits(DenseDesignMatrix):
    
    def __init__(self, which_set, start=None, stop=None, preprocessor=None, axes=['c', 0, 1, 'b']):
        assert which_set in ['train','test']
        
        X = np.load(os.path.join(DATA_DIR,which_set+'.npy'))
        X = np.cast['float32'](X)
        
        if which_set == 'test':
            # dummy targets
            y = np.zeros((X.shape[0],10))
        else:
            y = np.load(os.path.join(DATA_DIR,'targets.npy'))
            one_hot = np.zeros((y.shape[0],10),dtype='float32')
            for i in xrange(y.shape[0]):
                one_hot[i,int(y[i])] = 1.
            y = one_hot
        
        def dimshuffle(b01c):
            default = ('b', 0, 1, 'c')
            return b01c.transpose(*[default.index(axis) for axis in axes])
        
        if start is not None:
            assert start >= 0
            assert stop > start
            assert stop <= X.shape[0]
            X = X[start:stop, :, :]
            y = y[start:stop]
            assert X.shape[0] == y.shape[0]
        
        topo_view = X
        m, r, c = topo_view.shape
        assert r == 28
        assert c == 28
        topo_view = topo_view.reshape(m,r,c,1)
        
        super(Digits,self).__init__(topo_view = dimshuffle(topo_view), y=y, axes=axes)
        
        assert not np.any(np.isnan(self.X))

def get_dataset(tot=False, preprocessor='normal'):
    if not os.path.exists(DATA_DIR+'train.npy') or \
        not os.path.exists(DATA_DIR+'test.npy') or \
        not os.path.exists(DATA_DIR+'targets.npy'):
        initial_read()
    
    train_path = DATA_DIR+'train_'+preprocessor+'_preprocessed.pkl'
    valid_path = DATA_DIR+'valid_'+preprocessor+'_preprocessed.pkl'
    tottrain_path = DATA_DIR+'tottrain_'+preprocessor+'_preprocessed.pkl'
    test_path = DATA_DIR+'test_'+preprocessor+'_preprocessed.pkl'
    
    if os.path.exists(train_path) and os.path.exists(valid_path) and os.path.exists(test_path):
        
        print 'loading preprocessed data'
        trainset = serial.load(train_path)
        validset = serial.load(valid_path)
        if tot:
            tottrainset = serial.load(tottrain_path)
        testset = serial.load(test_path)
    else:
        
        print 'loading raw data...'
        trainset = Digits(which_set='train', start=0, stop=34000)
        validset = Digits(which_set='train', start=34000, stop=42000)
        tottrainset = Digits(which_set='train')
        testset = Digits(which_set='test')
        
        print 'preprocessing data...'
        pipeline = preprocessing.Pipeline()
        pipeline.items.append(preprocessing.GlobalContrastNormalization(sqrt_bias=10., use_std=True))
        
        if preprocessor != 'nozca':
            # ZCA = zero-phase component analysis
            # very similar to PCA, but preserves the look of the original image better
            pipeline.items.append(preprocessing.ZCA())
        
        # note the can_fit=False's: no sharing between train and valid data
        trainset.apply_preprocessor(preprocessor=pipeline, can_fit=True)
        validset.apply_preprocessor(preprocessor=pipeline, can_fit=False)
        tottrainset.apply_preprocessor(preprocessor=pipeline, can_fit=True)
        testset.apply_preprocessor(preprocessor=pipeline, can_fit=False)
        
        if preprocessor not in ('normal','nozca'):
            for data in (trainset, validset, tottrainset, testset):
                for ii in range(data.X.shape[0]):
                    # normalize to [0,1]
                    dmax = np.max(data.X[ii,:])
                    dmin = np.min(data.X[ii,:])
                    dnorm = (data.X[ii,:] - dmin) / (dmax - dmin)
                    # and convert to PIL image
                    img = Image.fromarray(dnorm.reshape(28, 28) * 255.).convert('L')
                    
                    # apply preprocessor
                    if preprocessor == 'rotate':
                        rot = rng.randint(0, 360)
                        img = img.rotate(rot, Image.BILINEAR)
                    elif preprocessor == 'emboss':
                        img = emboss(img)
                    elif preprocessor == 'hshear':
                        # coef = 0 means unsheared
                        coef = -1 + np.random.rand()*2
                        # note: image is moved with (coef/2)*28 to center it after shearing
                        img = img.transform((28,28), Image.AFFINE, (1,coef,-(coef/2)*28,0,1,0), Image.BILINEAR)
                    elif preprocessor == 'vshear':
                        coef = -1 + np.random.rand()*2
                        img = img.transform((28,28), Image.AFFINE, (1,0,0,coef,1,-(coef/2)*28), Image.BILINEAR)
                    elif preprocessor == 'patch':
                        x1 = np.random.randint(-6, 6)
                        y1 = np.random.randint(-6, 6)
                        x2 = np.random.randint(-6, 6)
                        y2 = np.random.randint(-6, 6)
                        img = img.transform((28,28), Image.EXTENT, (x1, y1, 28+x2, 28+y2), Image.BILINEAR)
                    
                    # convert back to numpy array
                    data.X[ii,:] = np.array(img.getdata()) / 255.
                    
                    if preprocessor == 'noisy':
                        # add noise
                        data.X[ii,:] += np.random.randn(28*28) * 0.25
                        # bound between [0,1]
                        data.X[ii,:] = np.minimum(np.ones(28*28), np.maximum(np.zeros(28*28), data.X[ii,:]))
        
        # this uses numpy format for storage instead of pickle, for memory reasons
        trainset.use_design_loc(DATA_DIR+'train_'+preprocessor+'_design.npy')
        validset.use_design_loc(DATA_DIR+'valid_'+preprocessor+'_design.npy')
        tottrainset.use_design_loc(DATA_DIR+'tottrain_'+preprocessor+'_design.npy')
        testset.use_design_loc(DATA_DIR+'test_'+preprocessor+'_design.npy')
        # this path can be used for visualizing weights after training is done
        trainset.yaml_src = '!pkl: "%s"' % train_path
        validset.yaml_src = '!pkl: "%s"' % valid_path
        tottrainset.yaml_src = '!pkl: "%s"' % tottrain_path
        testset.yaml_src = '!pkl: "%s"' % test_path
        
        print 'saving preprocessed data...'
        serial.save(train_path, trainset)
        serial.save(valid_path, validset)
        serial.save(tottrain_path, tottrainset)
        serial.save(test_path, testset)
        
    if tot:
        return tottrainset, validset, testset
    else:
        return trainset, validset, testset

def emboss(img):
    
    azi = rng.randint(0, 360)
    ele = rng.randint(0, 60)
    dep = 2
    
    # defining azimuth, elevation, and depth
    ele = (ele * 2 * np.pi) / 360.
    azi = (azi * 2 * np.pi) / 360.

    a = np.asarray(img).astype('float')
    # find the gradient
    grad = np.gradient(a)
    # (it is two arrays: grad_x and grad_y)
    grad_x, grad_y = grad
    # getting the unit incident ray
    gd = np.cos(ele) # length of projection of ray on ground plane
    dx = gd * np.cos(azi)
    dy = gd * np.sin(azi)
    dz = np.sin(ele)
    # adjusting the gradient by the "depth" factor
    # (I think this is how GIMP defines it)
    grad_x = grad_x * dep / 100.
    grad_y = grad_y * dep / 100.
    # finding the unit normal vectors for the image
    leng = np.sqrt(grad_x**2 + grad_y**2 + 1.)
    uni_x = grad_x/leng
    uni_y = grad_y/leng
    uni_z = 1./leng
    # take the dot product
    a2 = 255 * (dx*uni_x + dy*uni_y + dz*uni_z)
    # avoid overflow
    a2 = a2.clip(0, 255)
    # you must convert back to uint8 /before/ converting to an image
    return Image.fromarray(a2.astype('uint8'))

if __name__ == '__main__':
    pass
