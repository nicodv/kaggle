#!/usr/bin/python

import os
import numpy as np
import pandas as pd
from theano import function
import WhaleRedux.whaledata

from pylearn2.train import Train
from pylearn2.models.mlp import MLP, ConvRectifiedLinear, Softmax
from pylearn2.costs.mlp.dropout import Dropout
from pylearn2.space import Conv2DSpace
from pylearn2.training_algorithms.sgd import SGD, ExponentialDecay, MomentumAdjustor
from pylearn2.termination_criteria import EpochCounter
from sklearn.metrics.metrics import auc_score

DATA_DIR = '/home/nico/datasets/Kaggle/WhaleRedux/'


def get_conv2D(dim_input, batch_size=200):
    config = {
        'batch_size': batch_size,
        'input_space': Conv2DSpace(shape=dim_input[:2], num_channels=dim_input[2]),
        'layers': [
        ConvRectifiedLinear(layer_name='h0', output_channels=30, irange=.04, init_bias=0., max_kernel_norm=1.9365,
            kernel_shape=[5, 5], border_mode = 'valid', pool_shape=[4, 4], pool_stride=[2, 2], W_lr_scale=0.64),
        ConvRectifiedLinear(layer_name='h1', output_channels=40, irange=.05, init_bias=0., max_kernel_norm=1.9365,
            kernel_shape=[3, 3], pool_shape=[4, 4], pool_stride=[4, 4], W_lr_scale=1.),
        ConvRectifiedLinear(layer_name='h2', output_channels=50, irange=.05, init_bias=0., max_kernel_norm=1.9365,
            kernel_shape=[3, 3], pool_shape=[2, 2], pool_stride=[2, 2], W_lr_scale=1.),
        Softmax(layer_name='y', n_classes=2, istdev=.025, W_lr_scale=0.25)
        ]
    }
    return MLP(**config)

def get_conv1D(dim_input, batch_size=200):
    config = {
        'batch_size': batch_size,
        'input_space': Conv2DSpace(shape=dim_input[:2], num_channels=dim_input[2]),
        'layers': [
        ConvRectifiedLinear(layer_name='h0', output_channels=30, irange=.04, init_bias=0.5, max_kernel_norm=1.9365,
            kernel_shape=[7, 1], pool_shape=[4, 1], pool_stride=[3, 1], W_lr_scale=0.64),
        ConvRectifiedLinear(layer_name='h1', output_channels=40, irange=.05, init_bias=0., max_kernel_norm=1.9365,
            kernel_shape=[5, 1], pool_shape=[4, 1], pool_stride=[1, 1], W_lr_scale=1.),
        ConvRectifiedLinear(layer_name='h2', output_channels=50, irange=.05, init_bias=0., max_kernel_norm=1.9365,
            kernel_shape=[5, 1], pool_shape=[4, 1], pool_stride=[1, 1], W_lr_scale=1.),
        ConvRectifiedLinear(layer_name='h3', output_channels=60, irange=.05, init_bias=0., max_kernel_norm=1.9365,
            kernel_shape=[3, 1], pool_shape=[4, 1], pool_stride=[2, 1], W_lr_scale=1.),
        Softmax(layer_name='y', n_classes=2, irange=.025, W_lr_scale=0.25)
        ]
    }
    return MLP(**config)

def get_trainer(model, trainset, validset, epochs=20, batch_size=200):
    monitoring_batches = None if validset is None else 20
    train_algo = SGD(
        batch_size = batch_size,
        init_momentum = 0.5,
        learning_rate = 0.1,
        monitoring_batches = monitoring_batches,
        monitoring_dataset = validset,
        cost = Dropout(input_include_probs={'h0': 0.8, 'h1': 1., 'h2': 1., 'y': 0.5},
                        input_scales={'h0': 1./0.8, 'h1': 1./1., 'h2': 1./1., 'y': 1./0.5},
                        default_input_include_prob=0.5, default_input_scale=1./0.5),
        termination_criterion = EpochCounter(epochs),
        update_callbacks = ExponentialDecay(decay_factor=1.0005, min_lr=0.001)
    )
    return Train(model=model, algorithm=train_algo, dataset=trainset, save_freq=0, save_path='epoch', \
            extensions=[MomentumAdjustor(final_momentum=0.9, start=0, saturate=int(epochs*0.8)), ])

def get_output(model, tdata, layerindex, batch_size=100):
    # get output submodel classifiers
    Xb = model.get_input_space().make_theano_batch()
    Yb = model.fprop(Xb, return_all=True)
    
    data = tdata.get_topological_view()
    # fill up with zeroes for dividible by batch number
    extralength = batch_size - data.shape[0]%batch_size
    
    if extralength < batch_size:
        data = np.append(data,np.zeros([extralength, data.shape[1],data.shape[2],data.shape[3]]), axis=0)
        data = data.astype('float32')
    
    propagate = function([Xb], Yb, allow_input_downcast=True)
    
    output = []
    for ii in xrange(int(data.shape[0]/batch_size)):
        seldata = data[ii*batch_size:(ii+1)*batch_size,:]
        output.append(propagate(seldata)[layerindex])
    
    output = np.reshape(output,[data.shape[0],-1])
    
    if extralength < batch_size:
        # remove the filler
        output = output[:-extralength]
    
    return output


if __name__ == '__main__':
    
    submission = True
    batch_size = 50
    
    ####################
    #   MEL SPECTRUM   #
    ####################
    trainset,validset,testset = WhaleRedux.whaledata.get_dataset('melspectrum', tot=submission)
    
    # build and train classifiers for submodels
    model = get_conv2D([16,16,1], batch_size=batch_size)
    get_trainer(model, trainset, validset, epochs=20, batch_size=batch_size).main_loop()
    
    # validate model
    if not submission:
        output = get_output(model, validset, -1)
        # calculate AUC using sklearn
        AUC = auc_score(validset.get_targets()[:,0],output[:,0])
        print AUC
    else:
        fn = np.load(os.path.join(DATA_DIR,'filenames.npy'))
        outtestset = get_output(model,testset,-1)[:,0]
        
        output = pd.DataFrame({'clip': fn, 'probability': outtestset})
        # save test output as submission
        output.to_csv(DATA_DIR+'model_conv2net.csv', header=True, index=False)
        
        # construct data sets with model output
        outtrainset = get_output(model,trainset,-2)
        outtestset = get_output(model,testset,-2)
        
        np.save(DATA_DIR+'conv2out_train', outtrainset)
        np.save(DATA_DIR+'conv2out_test', outtestset)
    
    
    #########################
    #   SPECTRAL FEATURES   #
    #########################
    trainset2,validset2,testset2 = WhaleRedux.whaledata.get_dataset('specfeat', tot=submission)
    
    # build and train classifiers for submodels
    model2 = get_conv1D([16,1,24], batch_size=batch_size)
    get_trainer(model2, trainset2, validset2, epochs=40, batch_size=batch_size).main_loop()
    
    # validate model
    if not submission:
        output = get_output(model2,validset2)
        # calculate AUC using sklearn
        AUC = auc_score(validset2.get_targets()[:,0],output[:,0])
        print AUC
    else:
        fn = np.load(os.path.join(DATA_DIR,'filenames.npy'))
        outtestset2 = get_output(model2,testset2,-1)[:,0]
        
        output2 = pd.DataFrame({'clip': fn, 'probability': outtestset2})
        # save test output as submission
        output2.to_csv(DATA_DIR+'model_conv1net.csv', header=True, index=False)
        
        outtrainset = get_output(model2,trainset2,-2)
        outtestset = get_output(model2,testset2,-2)
        
        np.save(DATA_DIR+'conv1out_train', outtrainset)
        np.save(DATA_DIR+'conv1out_test', outtestset)
    
