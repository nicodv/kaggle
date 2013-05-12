#!/usr/bin/python

import os
import pickle
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
import pylearn2.models.autoencoder as autoencoder
import pylearn2.models.sparse_autoencoder as sparse_autoencoder
from pylearn2.models.softmax_regression import SoftmaxRegression

from pylearn2.train import Train
import pylearn2.training_algorithms.sgd as sgd
from pylearn2.termination_criteria import EpochCounter
import pylearn2.costs as costs
from pylearn2.costs.autoencoder import MeanBinaryCrossEntropy, MeanSquaredReconstructionError
import pylearn2.costs.mlp.dropout as dropout
from pylearn2.corruption import GaussianCorruptor, BinomialCorruptor
from pylearn2.costs.ebm_estimation import SMD, NCE
from pylearn2.distributions.mnd import MND

DATA_DIR = '/home/nico/datasets/Kaggle/BlackBox/'

def process_data():
    # pre-process unsupervised data
    if not os.path.exists(DATA_DIR+'preprocess.pkl') \
    or not os.path.exists(DATA_DIR+'unsup_prep_data.pkl') \
    or not os.path.exists(DATA_DIR+'sup_prep_data.pkl'):
        unsup_data = black_box_dataset.BlackBoxDataset('extra')
        pipeline = preprocessing.Pipeline()
        pipeline.items.append(preprocessing.Standardize(global_mean=False, global_std=False))
        #pipeline.items.append(preprocessing.ZCA(filter_bias=.1))
        unsup_data.apply_preprocessor(preprocessor=pipeline, can_fit=True)
        serial.save(DATA_DIR+'preprocess.pkl', pipeline)
        
        # why the hell do I get pickling errors if I use serial here? solve by pickling myself
        #serial.save(DATA_DIR+'unsup_prep_data.pkl', unsup_data)
        out = open(DATA_DIR+'unsup_prep_data.pkl', 'w')
        pickle.dump(unsup_data, out)
        out.close()
        
        # process supervised training data
        sup_data = []
        which_data = ['train']*3 + ['public_test']
        starts = [0, 800, None, None]
        stops = [800, 1000, None, None]
        fits = [False, False, False, False]
        for curstr, start, stop, fit in zip(which_data, starts, stops, fits):
            sup_data.append(black_box_dataset.BlackBoxDataset(
            which_set=curstr,
            start=start,
            stop=stop,
            preprocessor=pipeline,
            fit_preprocessor=fit
            ))
        serial.save(DATA_DIR+'sup_prep_data.pkl', sup_data)
        
    else:
        pipeline = serial.load(DATA_DIR+'preprocess.pkl')
        #unsup_data = serial.load(DATA_DIR+'unsup_prep_data.pkl')
        unsup_data = pickle.load(open(DATA_DIR+'unsup_prep_data.pkl', 'r'))
        sup_data = serial.load(DATA_DIR+'sup_prep_data.pkl')
    
    return unsup_data, sup_data

def construct_stacked_rbm(structure):
    # some RBM-universal settings
    irange = 0.5
    init_bias = 0.
    
    grbm = rbm.GaussianBinaryRBM(
        nvis=structure[0],
        nhid=structure[1],
        irange=irange,
        energy_function_class=GRBM_Type_1,
        learn_sigma=False,
        init_sigma=1.,
        init_bias_hid=init_bias,
        mean_vis=False,
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

def construct_ae(structure):
    # some settings
    irange = 0.05
    
    layers = []
    for vsize,hsize in zip(structure[:-1], structure[1:]):
        layers.append(autoencoder.DenoisingAutoencoder(
            corruptor=BinomialCorruptor(0.5),
            nvis=vsize,
            nhid=hsize,
            tied_weights=True,
            act_enc='sigmoid',
            act_dec='sigmoid',
            irange=irange
        ))
    return StackedBlocks(layers)

def construct_dbn_from_stack(stack):
    # some settings
    irange = 0.05
    
    layers = []
    for ii, layer in enumerate(stack.layers()):
        lr_scale = 1. if ii==0 else 0.25
        layers.append(mlp.Sigmoid(
            dim=layer.nhid,
            layer_name='h'+str(ii),
            irange=irange,
            W_lr_scale=lr_scale,
            max_col_norm=2.
        ))
    # softmax layer at then end for classification
    layers.append(mlp.Softmax(
        #nvis=stackedrbm.layers()[-1].nhid,
        n_classes=9,
        layer_name='y',
        irange=irange,
        W_lr_scale=0.25
    ))
    dbn = mlp.MLP(layers=layers, nvis=stack.layers()[0].get_input_space().dim)
    # copy weigths to DBN
    for ii, layer in enumerate(stack.layers()):
        dbn.layers[ii].set_weights(layer.get_weights())
        if isinstance(layer, rbm.RBM):
            dbn.layers[ii].set_biases(layer.bias_hid.get_value(borrow=False))
        elif isinstance(layer, autoencoder.Autoencoder):
            dbn.layers[ii].set_biases(layer.hidbias.get_value(borrow=False))
    return dbn

def get_ae_pretrainer(layer, data, batch_size):
    init_lr = 0.05
    dec_fac = 1.0001
    
    train_algo = sgd.SGD(
        batch_size = batch_size,
        learning_rate = init_lr,
        init_momentum = 0.5,
        monitoring_batches = 100/batch_size,
        monitoring_dataset = {'train': data},
        cost = MeanSquaredReconstructionError(),
        termination_criterion =  EpochCounter(20),
        update_callbacks = sgd.ExponentialDecay(decay_factor=dec_fac, min_lr=0.001)
        )
    return Train(model=layer, algorithm=train_algo, dataset=data, \
            extensions=[sgd.MomentumAdjustor(final_momentum=0.9, start=0, saturate=5), ])

def get_rbm_pretrainer(layer, data, batch_size):
    # GBRBM needs smaller learning rate for stability?
    if isinstance(layer, rbm.GaussianBinaryRBM):
        init_lr = 0.5
        dec_fac = 1.000005
    else:
        init_lr = 0.5
        dec_fac = 1.000005
    
    train_algo = sgd.SGD(
        batch_size = batch_size,
        learning_rate = init_lr,
        init_momentum = 0.5,
        monitoring_batches = 100/batch_size,
        monitoring_dataset = {'train': data},
        cost = SMD(GaussianCorruptor(0.5)),
        termination_criterion =  EpochCounter(5),
        update_callbacks = sgd.ExponentialDecay(decay_factor=dec_fac, min_lr=0.001)
        )
    return Train(model=layer, algorithm=train_algo, dataset=data, \
            extensions=[sgd.MomentumAdjustor(final_momentum=0.9, start=0, saturate=5), ])

def get_finetuner(model, trainset, validset=None, batch_size=100):
    train_algo = sgd.SGD(
        batch_size = batch_size,
        init_momentum = 0.5,
        learning_rate = 0.5,
        monitoring_batches = 100/batch_size,
        monitoring_dataset = {'train': trainset, 'valid': validset},
        cost = dropout.Dropout(input_include_probs={'h0': 0.8}, input_scales={'h0': 1./0.8}),
        termination_criterion = EpochCounter(200),
        update_callbacks = sgd.ExponentialDecay(decay_factor=1.0001, min_lr=0.001)
    )
    return Train(model=model, algorithm=train_algo, dataset=trainset, save_freq=0, \
            extensions=[sgd.MomentumAdjustor(final_momentum=0.9, start=0, saturate=150), ])

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
    for ii in xrange(data.X.shape[0] // batch_size):
        x_arg = data.X[ii*batch_size:(ii+1)*batch_size,:]
        if Xb.ndim > 2:
            x_arg = data.get_topological_view(x_arg)
        y.append(f(x_arg.astype(Xb.dtype)))
    
    y = np.concatenate(y)
    assert y.ndim == 1
    assert y.shape[0] == data.X.shape[0]
    # discard any zero-padding that was used to give the batches uniform size
    y = y[:m]
    data.X = data.X[:m]
    
    return y

if __name__ == '__main__':
    
    # some settings
    submission = True
    structure = [1875, 3000, 3000, 2000, 2000, 2000]
    batch_size = 100
    
    unsup_data, sup_data = process_data()
    
    stack = construct_ae(structure)
    #stack = serial.load(DATA_DIR+'gbrbm_pretrained.pkl')
    
    # pre-train model
    for ii, layer in enumerate(stack.layers()):
        utraindata = unsup_data if ii==0 else TransformerDataset(raw=unsup_data,
                                                transformer=StackedBlocks(stack.layers()[:ii]))
        trainer = get_ae_pretrainer(layer, utraindata, batch_size)
        trainer.main_loop()
    
    # construct DBN
    dbn = construct_dbn_from_stack(stack)
    
    # train DBN
    if submission:
        traindata = sup_data[2]
        validdata = sup_data[2]
    else:
        traindata = sup_data[0]
        validdata = sup_data[1]
    
    finetuner = get_finetuner(dbn, traindata, validdata, batch_size)
    finetuner.main_loop()
    
    if submission:
        # get output
        output = get_output(dbn, sup_data[3], batch_size)
        
        # create submission
        out = open(DATA_DIR+'submission.csv', 'w')
        for i in xrange(output.shape[0]):
            out.write('%d.0\n' % (output[i] + 1))
        out.close()
    
