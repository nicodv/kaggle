#!/usr/bin/python

import numpy as np
from theano import function
import Whales.whaledata

from pylearn2.train import Train
from pylearn2.models.mlp import MLP, ConvRectifiedLinear, Softmax, RectifiedLinear, Sigmoid
from pylearn2.space import Conv2DSpace
from pylearn2.training_algorithms.sgd import SGD, ExponentialDecay, MomentumAdjustor
from pylearn2.termination_criteria import EpochCounter
from pylearn2.costs.cost import MethodCost
from sklearn.metrics.metrics import auc_score

DATA_DIR = '/home/nico/datasets/Kaggle/Whales/'

def get_conv2D(dim_input):
    config = {
        'batch_size': 100,
        'input_space': Conv2DSpace(shape=dim_input[:2], num_channels=dim_input[2]),
        'dropout_include_probs': [1, 1, 1, 0.5, 1],
        'dropout_input_include_prob': 0.8,
        'layers': [
        ConvRectifiedLinear(layer_name='h0', output_channels=10, irange=.04, init_bias=0.5,
            kernel_shape=[7, 7], pool_shape=[6, 4], pool_stride=[3, 2], W_lr_scale=0.64, border_mode='full'),
        ConvRectifiedLinear(layer_name='h1', output_channels=20, irange=.05, init_bias=0.,
            kernel_shape=[5, 5], pool_shape=[4, 4], pool_stride=[2, 2], W_lr_scale=1.),
        ConvRectifiedLinear(layer_name='h2', output_channels=40, irange=.05, init_bias=0.,
            kernel_shape=[3, 3], pool_shape=[4, 4], pool_stride=[2, 2], W_lr_scale=1.),
        Sigmoid(dim=100, layer_name='h3', irange=.05, W_lr_scale=1., init_bias=0.),
        Softmax(layer_name='y', n_classes=2, irange=.025, W_lr_scale=.25)
        ]
    }
    return MLP(**config)

def get_conv1D(dim_input):
    config = {
        'batch_size': 100,
        'input_space': Conv2DSpace(shape=dim_input[:2], num_channels=dim_input[2]),
        'dropout_include_probs': [1, 1, 1, 0.5, 1],
        'dropout_input_include_prob': 0.8,
        'layers': [
        ConvRectifiedLinear(layer_name='h0', output_channels=10, irange=.04, init_bias=0.5,
            kernel_shape=[7, 1], pool_shape=[6, 1], pool_stride=[3, 1], W_lr_scale=0.64, border_mode='full'),
        ConvRectifiedLinear(layer_name='h1', output_channels=20, irange=.05, init_bias=0.,
            kernel_shape=[5, 1], pool_shape=[4, 1], pool_stride=[2, 1], W_lr_scale=1.),
        ConvRectifiedLinear(layer_name='h2', output_channels=40, irange=.05, init_bias=0.,
            kernel_shape=[3, 1], pool_shape=[4, 1], pool_stride=[2, 1], W_lr_scale=1.),
        Sigmoid(dim=50, layer_name='h3', irange=.05, W_lr_scale=1., init_bias=0.),
        Softmax(layer_name='y', n_classes=2, irange=.025, W_lr_scale=.25)
        ]
    }
    return MLP(**config)

def get_trainer(model, trainset, validset, epochs=100):
    monitoring_batches = None if validset is None else 100
    train_algo = SGD(
        batch_size = 100,
        init_momentum = 0.5,
        learning_rate = 0.5,
        monitoring_batches = monitoring_batches,
        monitoring_dataset = validset,
        cost = MethodCost(method='cost_from_X', supervised=1),
        termination_criterion = EpochCounter(epochs),
        update_callbacks = ExponentialDecay(decay_factor=1.0005, min_lr=0.001)
    )
    return Train(model=model, algorithm=train_algo, dataset=trainset, save_freq=0, save_path='epoch', \
            extensions=[MomentumAdjustor(final_momentum=0.95, start=0, saturate=40), ])

def get_output(model, tdata, batch_size=100):
    # get output submodel classifiers
    Xb = model.get_input_space().make_theano_batch()
    Yb = model.fprop(Xb)
    
    data = tdata.get_topological_view()
    # fill up with zeroes for dividible by batch number
    extralength = batch_size - data.shape[0]%batch_size
    
    if extralength < 100:
        data = np.append(data,np.zeros([extralength, data.shape[1],data.shape[2],data.shape[3]]), axis=0)
        data = data.astype('float32')
    
    propagate = function([Xb], Yb)
    
    output = []
    for ii in xrange(int(data.shape[0]/batch_size)):
        seldata = data[ii*batch_size:(ii+1)*batch_size,:]
        output.append(propagate(seldata))
    
    output = np.reshape(output,[data.shape[0],-1])
    
    if extralength < 100:
        # remove the filler
        output = output[:-extralength]
    
    return output


if __name__ == '__main__':
    
    submission = False
    
    ########################
    #   LOG MEL SPECTRUM   #
    ########################
    trainset,validset,testset = Whales.whaledata.get_dataset('melspectrum', tot=submission)
    
    # build and train classifiers for submodels
    model = get_conv2D([67,40,1])
    get_trainer(model, trainset, validset).main_loop()
    
    # validate model
    if not submission:
        output = get_output(model,validset)
        # calculate AUC using sklearn
        AUC = auc_score(validset.get_targets()[:,0],output[:,0])
        print AUC
    else:
        # construct data sets with model output
        del model.layers[-1]
        del model.dropout_include_probs[-1]
        del model.dropout_scales[-1]
        outtrainset = get_output(model,trainset)
        outtestset = get_output(model,testset)
        # save test output as submission
        np.savetxt(DATA_DIR+'model_convnet.csv', outtestset[:,0], delimiter=",")
        
        # reshape
        train_no = outtrainset.shape[0]
        test_no = outtestset.shape[0]
        outtrainset = np.swapaxes(np.reshape(outtrainset,[train_no,outtrainset.shape[1]]),0,1)
        outtrainset = np.reshape(outtrainset,[train_no,-1])
        outtestset = np.swapaxes(np.reshape(outtestset,[test_no,outtestset.shape[1]]),0,1)
        outtestset = np.reshape(outtestset,[test_no,-1])
        
        np.save(DATA_DIR+'convout_train', outtrainset)
        np.save(DATA_DIR+'convout_test', outtestset)
    
    
    #########################
    #   SPECTRAL FEATURES   #
    #########################
    trainset2,validset2,testset2 = Whales.whaledata.get_dataset('specfeat', tot=submission)
    
    # build and train classifiers for submodels
    model2 = get_conv1D([67,1,24])
    get_trainer(model2, trainset2, validset2, 50).main_loop()
    
    # validate model
    if not submission:
        output = get_output(model2,validset2)
        # calculate AUC using sklearn
        AUC = auc_score(validset2.get_targets()[:,0],output[:,0])
        print AUC
    else:
        # construct data sets with model output
        del model2.layers[-1]
        del model2.dropout_include_probs[-1]
        del model2.dropout_scales[-1]
        outtrainset = get_output(model2,trainset2)
        outtestset = get_output(model2,testset2)
        # save test output as submission
        np.savetxt(DATA_DIR+'model_conv1net.csv', outtestset[:,0], delimiter=",")
        
        # reshape
        train_no = outtrainset.shape[0]
        test_no = outtestset.shape[0]
        outtrainset = np.swapaxes(np.reshape(outtrainset,[train_no,outtrainset.shape[1]]),0,1)
        outtrainset = np.reshape(outtrainset,[train_no,-1])
        outtestset = np.swapaxes(np.reshape(outtestset,[test_no,outtestset.shape[1]]),0,1)
        outtestset = np.reshape(outtestset,[test_no,-1])
        
        np.save(DATA_DIR+'conv1out_train', outtrainset)
        np.save(DATA_DIR+'conv1out_test', outtestset)
    
