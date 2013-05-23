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

from pylearn2.base import StackedBlocks
import pylearn2.models.mlp as mlp
import pylearn2.models.autoencoder as autoencoder

from pylearn2.train import Train
import pylearn2.training_algorithms.sgd as sgd
from pylearn2.termination_criteria import EpochCounter
import pylearn2.costs as costs
from pylearn2.costs.autoencoder import MeanBinaryCrossEntropy, MeanSquaredReconstructionError
import pylearn2.costs.mlp.dropout as dropout
from pylearn2.corruption import GaussianCorruptor, BinomialCorruptor

DATA_DIR = '/home/nico/datasets/Kaggle/BlackBox/'

class CAE_cost(costs.cost.Cost):
    def __call__(self, model, X, Y=None, coef1=0.05, coef2=0, ** kwargs):
        if coef1 and coef2:
            cost = ((model.reconstruct(X) - X) ** 2).sum(axis=1).mean() + coef1*model.contraction_penalty(X) + coef2*model.higher_order_penalty(X)
        elif coef1:
            cost = ((model.reconstruct(X) - X) ** 2).sum(axis=1).mean() + coef1*model.contraction_penalty(X)
        else:
            cost = ((model.reconstruct(X) - X) ** 2).sum(axis=1).mean()
        return cost

class Rectify(object):
    def __call__(self, X_before_activation):
        # X_before_activation is linear inputs of hidden units, dense
        return X_before_activation * (X_before_activation > 0)

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

def construct_ae(structure):
    # some settings
    irange = 0.05
    
    layers = []
    for vsize,hsize in zip(structure[:-1], structure[1:]):
        # DenoisingAutoencoder?, ContractiveAutoencoder?, HigherOrderContractiveAutoencoder?
        layers.append(autoencoder.ContractiveAutoencoder(
            # DenoisingAutoencoder
            #corruptor=BinomialCorruptor(0.6),
            # HigherOrderContractiveAutoencoder
            #corruptor=GaussianCorruptor(0.5),
            #num_corruptions=8,
            nvis=vsize,
            nhid=hsize,
            tied_weights=True,
            act_enc='sigmoid',
            act_dec='sigmoid',
            #act_enc=Rectify(),
            #act_dec=Rectify(),
            irange=irange
        ))
    return StackedBlocks(layers)

def construct_dbn_from_stack(stack, dropout_strategy='default'):
    # some settings
    irange = 0.05
    
    layers = []
    for ii, layer in enumerate(stack.layers()):
        if ii==0 or dropout_strategy=='default':
            lr_scale = 0.16
        elif ii==1:
            lr_scale = 0.16
        elif ii==2:
            lr_scale = 0.25
        elif ii==3:
            lr_scale = 0.36
        elif ii==4:
            lr_scale = 0.49
        elif ii==5:
            lr_scale = 0.64
        layers.append(mlp.Sigmoid(
            dim=layer.nhid,
            layer_name='h'+str(ii),
            irange=irange,
            W_lr_scale=lr_scale,
            max_col_norm=2.
        ))
    # softmax layer at then end for classification
    layers.append(mlp.Softmax(
        n_classes=9,
        layer_name='y',
        irange=irange,
        W_lr_scale=0.16
    ))
    dbn = mlp.MLP(layers=layers, nvis=stack.layers()[0].get_input_space().dim)
    # copy weigths to DBN
    for ii, layer in enumerate(stack.layers()):
        dbn.layers[ii].set_weights(layer.get_weights())
        dbn.layers[ii].set_biases(layer.hidbias.get_value(borrow=False))
    return dbn

def get_ae_pretrainer(layer, data, batch_size):
    init_lr = 0.1
    dec_fac = 1.0001
    
    train_algo = sgd.SGD(
        batch_size = batch_size,
        learning_rate = init_lr,
        init_momentum = 0.5,
        monitoring_batches = 100/batch_size,
        monitoring_dataset = {'train': data},
        #cost = MeanSquaredReconstructionError(),
        cost = CAE_cost(),
        termination_criterion =  EpochCounter(20),
        update_callbacks = sgd.ExponentialDecay(decay_factor=dec_fac, min_lr=0.02)
        )
    return Train(model=layer, algorithm=train_algo, dataset=data, \
            extensions=[sgd.MomentumAdjustor(final_momentum=0.9, start=0, saturate=5), ])

def get_finetuner(model, trainset, validset=None, batch_size=100, dropout_strategy='default'):
    if dropout_strategy == 'default':
        cost = dropout.Dropout(input_include_probs={'h0': 0.4}, input_scales={'h0': 1./0.4}, 
                               default_input_include_prob=0.4, default_input_scale=1./0.4)
    elif dropout_strategy == 'fan':
        cost = dropout.Dropout(input_include_probs={'h0': 0.3, 'h1': 0.4, 'h2': 0.5, 'h3': 0.6, 'h4': 0.7, 'h5': 0.8, 'y': 0.9},
            input_scales={'h0': 1./0.3, 'h1': 1./0.4, 'h2': 1./0.5, 'h3': 1./0.6, 'h4': 1./0.7, 'h5': 1./0.8, 'y': 1./0.9})
    
    train_algo = sgd.SGD(
        batch_size = batch_size,
        init_momentum = 0.5,
        learning_rate = 0.5,
        monitoring_batches = 100/batch_size,
        monitoring_dataset = {'train': trainset, 'valid': validset},
        cost = cost,
        termination_criterion = EpochCounter(500),
        update_callbacks = sgd.ExponentialDecay(decay_factor=1.0005, min_lr=0.05)
    )
    return Train(model=model, algorithm=train_algo, dataset=trainset, save_freq=0, \
            extensions=[sgd.MomentumAdjustor(final_momentum=0.9, start=0, saturate=400), ])

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
    submission = False
    # note: MSE van CAE gaat omhoog bij 4e layer, maar bij 5e layer enorm omlaag (?)
    structure = [1875, 2000, 2000, 2000, 2000, 2000, 2000]
    batch_size = 100
    
    unsup_data, sup_data = process_data()
    
    stack = construct_ae(structure)
    stack = serial.load(DATA_DIR+'cae6_005_pretrained.pkl')
    
    # pre-train model
    #for ii, layer in enumerate(stack.layers()):
        #utraindata = unsup_data if ii==0 else TransformerDataset(raw=unsup_data,
        #                                        transformer=StackedBlocks(stack.layers()[:ii]))
        #pretrainer = get_ae_pretrainer(layer, utraindata, batch_size)
        #pretrainer.main_loop()
    
    #serial.save(DATA_DIR+'cae6_005_pretrained.pkl', stack)
    
    # construct DBN
    dbn = construct_dbn_from_stack(stack, dropout_strategy='default')
    
    # train DBN
    if submission:
        traindata = sup_data[2]
        validdata = sup_data[2]
    else:
        traindata = sup_data[0]
        validdata = sup_data[1]
    
    # total finetuner
    finetuner = get_finetuner(dbn, traindata, validdata, batch_size, dropout_strategy='default')
    finetuner.main_loop()
    
    if submission:
        # get output
        output = get_output(dbn, sup_data[3], batch_size)
        
        # create submission
        out = open(DATA_DIR+'submission.csv', 'w')
        for i in xrange(output.shape[0]):
            out.write('%d.0\n' % (output[i] + 1))
        out.close()
    
