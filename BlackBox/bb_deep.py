#!/usr/bin/python

import os
import numpy as np
from theano import function
from theano import tensor as T

from pylearn2.utils import serial
import BlackBox.black_box_dataset as black_box_dataset
import pylearn2.datasets.preprocessing as preprocessing
from pylearn2.datasets.transformer_dataset import TransformerDataset

import pylearn2.models.rbm as rbm
from pylearn2.energy_functions.rbm_energy import GRBM_Type_1
from pylearn2.base import StackedBlocks
import pylearn2.models.mlp as mlp
from pylearn2.models.softmax_regression import SoftmaxRegression

from pylearn2.train import Train
import pylearn2.training_algorithms.sgd as sgd
from pylearn2.termination_criteria import EpochCounter
import pylearn2.costs as costs
from pylearn2.costs.dbm import VariationalPCD, WeightDecay, TorontoSparsity

DATA_DIR = '/home/nico/datasets/Kaggle/BlackBox/'

def process_data():
    # pre-process unsupervised data
    if not os.path.exists(DATA_DIR+'preprocess.pkl') and os.path.exists(DATA_DIR+'unsup_prep_data.npy'):
        unsup_data = black_box_dataset.BlackBoxDataset('extra')
        pipeline = preprocessing.Pipeline()
        pipeline.items.append(preprocessing.Standardize(global_mean=False, global_std=False))
        pipeline.items.append(preprocessing.ZCA(filter_bias=.1))
        unsup_data.apply_preprocessor(preprocessor=pipeline, can_fit=True)
        serial.save(DATA_DIR+'preprocess.pkl', pipeline)
        np.save(DATA_DIR+'unsup_prep_data.npy', unsup_data)
    else:
        pipeline = serial.load(DATA_DIR+'preprocess.pkl')
        unsup_data = serial.load(DATA_DIR+'unsup_prep_data.npy')
    
    # process supervised training data
    sup_data = []
    which_data = ['train']*3 + ['public_test']
    starts = [0, 900, None, None]
    stops = [900, 1000, None, None]
    for curstr, start, stop in zip(which_data, starts, stops):
        sup_data.append(black_box_dataset.BlackBoxDataset(
        which_set=curstr,
        start=start,
        stop=stop,
        preprocessor=pipeline
        ))
    
    return unsup_data, sup_data

def construct_stacked_rbm(structure):
    # some RBM-universal settings
    irange = 0.01
    init_bias = 0.
    
    grbm = rbm.GaussianBinaryRBM(
        nvis=structure[0],
        nhid=structure[1],
        irange=irange,
        energy_function_class=GRBM_Type_1,
        learn_sigma=False,
        init_sigma=1.,
        init_bias_hid=init_bias,
        mean_vis=True,
        sigma_lr_scale=1.
    )
    rbms = []
    for vsize,hsize in zip(structure[1:-1], structure[2:]):
        rbms.append(rbm.RBM(
            nvis=vsize,
            nhid=hsize,
            irange=irange,
            init_bias_hid=init_bias
        ))
    return StackedBlocks([grbm] + rbms)

def construct_dbn(stackedrbm):
    layers = []
    for ii,rbm in enumerate(stackedrbm.layers):
        layers.append(mlp.RBM_Layer(
            layer_name='h'+ii,
            rbm=rbm
        ))
    layers.append(SoftmaxRegression(
        nvis=stackedrbm.layers[-1].nhid,
        n_classes=9,
        irange=0.05,
        W_lr_scale=0.25
    ))
    dbn = mlp.MLP(
        layers=layers,
        nvis=stackedrbm.layers[0].nvis
    )
    return dbn

def get_pretrainer(layer, data, batch_size):
    # GBRBM needs smaller learning rate for stability
    if isinstance(layer, rbm.GaussianBinaryRBM):
        init_lr = 0.001
    else:
        init_lr = 0.1
        
    train_algo = sgd.SGD(
        batch_size = batch_size,
        learning_rate = init_lr,
        init_momentum = 0.5,
        monitoring_batches = 100/batch_size,
        monitoring_dataset = data,
        cost = costs.cost.SumOfCosts(
            costs=[
                VariationalPCD(num_chains=100, num_gibbs_steps=5),
                WeightDecay(coeffs=[0.0001]),
                TorontoSparsity(targets=[0.2], coeffs=[0.001])
                ]
            ),
        termination_criterion =  EpochCounter(100),
        update_callbacks = sgd.ExponentialDecay(decay_factor=1.00005, min_lr=0.0001)
        )
    return Train(model=layer, algorithm=train_algo, dataset=data, \
            extensions=[sgd.MomentumAdjustor(final_momentum=0.9, start=0, saturate=80), ])

def get_finetuner(model, trainset, validset=None, batch_size=100):
    train_algo = sgd.SGD(
        batch_size = 100,
        init_momentum = 0.5,
        learning_rate = 0.5,
        monitoring_batches = 100/batch_size,
        monitoring_dataset = validset,
        cost = costs.mlp.dropout.Dropout(input_include_prob={'h0': 0.8}, input_scales={'h0': 1.}),
        termination_criterion = EpochCounter(250),
        update_callbacks = sgd.ExponentialDecay(decay_factor=1.0001, min_lr=0.001)
    )
    return Train(model=model, algorithm=train_algo, dataset=trainset, save_freq=0, \
            extensions=[sgd.MomentumAdjustor(final_momentum=0.9, start=0, saturate=200), ])

def get_output(model, data, batch_size):
    model.set_batch_size(batch_size)
    # dataset must be multiple of batch size of some batches will have
    # different sizes. theano convolution requires a hard-coded batch size
    m = data.X.shape[0]
    extra = batch_size - m % batch_size
    assert (m + extra) % batch_size == 0
    if extra > 0:
        data.X = np.concatenate((data.X, np.zeros((extra, data.X.shape[1]), dtype=data.X.dtype)), axis=0)
    assert data.X.shape[0] % batch_size == 0
    
    Xb = model.get_input_space().make_theano_batch()
    Yb = model.fprop(Xb)
    y = T.argmax(Yb, axis=1)
    f = function([Xb], y)
    
    y = []
    for ii in xrange(data.X.shape[0]/batch_size):
        x_arg = data[ii*batch_size:(ii+1)*batch_size,:]
        if Xb.ndim > 2:
            x_arg = data.get_topological_view(x_arg)
        y.append(f(x_arg.astype(Xb.dtype)))
    
    y = np.reshape(y,[data.shape[0],-1])
    y = np.concatenate(y)
    assert y.ndim == 1
    assert y.shape[0] == data.X.shape[0]
    # discard any zero-padding that was used to give the batches uniform size
    y = y[:m]
    
    return y

if __name__ == '__main__':
    
    # some settings
    submission = False
    structure = [1875, 500, 500]
    batch_size = 50
    
    unsup_data, sup_data = process_data()
    
    stackedrbm = construct_stacked_rbm(structure)
    
    # pre-train model
    for ii, layer in enumerate(stackedrbm.layers()):
        utraindata = TransformerDataset(raw=unsup_data, transformer=StackedBlocks(stackedrbm.layers()[:(ii+1)]))
        trainer = get_pretrainer(layer, utraindata, batch_size)
        trainer.main_loop()
    
    # construct DBN
    # (necessary to convert RBM layers to sigmoid layers for dropout?)
    dbn = construct_dbn(stackedrbm)
    
    # train DBN
    if submission:
        traindata = sup_data[0]
        validdata = sup_data[1]
    else:
        traindata = sup_data[2]
        validdata = sup_data[2]
    
    finetuner = get_finetuner(dbn, traindata, validdata, batch_size)
    finetuner.main_loop()
    
    # get output
    output = get_output(dbn, sup_data[3], batch_size)
    
    # create submission
    out = open(DATA_DIR+'submission.csv', 'w')
    for i in xrange(output.shape[0]):
        out.write('%d.0\n' % (output[i] + 1))
    out.close()
    
