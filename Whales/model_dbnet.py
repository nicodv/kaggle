#!/usr/bin/python

import os
import numpy as np
from theano import function
import Whales.whaledata

from pylearn2.train import Train
from pylearn2.models.mlp import MLP, Softmax, Sigmoid
from pylearn2.corruption import GaussianCorruptor
from pylearn2.costs.ebm_estimation import SMD
from pylearn2.models.rbm import RBM, GaussianBinaryRBM
from pylearn2.energy_functions.rbm_energy import GRBM_Type_1
from pylearn2.training_algorithms.sgd import SGD, ExponentialDecay, MomentumAdjustor
from pylearn2.termination_criteria import EpochCounter
from pylearn2.costs.cost import MethodCost
from pylearn2.base import StackedBlocks
from pylearn2.datasets.transformer_dataset import TransformerDataset
from sklearn.metrics.metrics import auc_score

DATA_DIR = '/home/nico/datasets/Kaggle/Whales/'


def get_grbm(structure):
    n_input, n_output = structure
    config = {
        'nvis': n_input,
        'nhid': n_output,
        "irange" : 0.05,
        "energy_function_class" : GRBM_Type_1,
        "learn_sigma" : False,
        "init_sigma" : 1.,
        "init_bias_hid" : -1.,
        "mean_vis" : True,
        "sigma_lr_scale" : 1.
        }
    return GaussianBinaryRBM(**config)

def get_rbm(structure):
    n_input, n_output = structure
    config = {
        'nvis': n_input,
        'nhid': n_output,
        "irange" : 0.05,
        "init_bias_hid" : -1.
        }
    return RBM(**config)

# hier is tegenwoordig de softmax regression-class voor
def get_classifier(n_inputs, nclasses=2):
    config = {
        'batch_size': 100,
        'nvis': n_inputs,
        'dropout_include_probs': [0.5, 1],
        'dropout_input_include_prob': 0.8,
        'layers': [
            Sigmoid(100, layer_name='h0', irange=.05, init_bias=-2.),
            Softmax(layer_name='y', n_classes=nclasses, istdev=.025, W_lr_scale=0.25),
        ]
    }
    return MLP(**config)

def get_rbmtrainer(layer, trainset):
    train_algo = SGD(
        batch_size = 100,
        learning_rate = 0.1,
        init_momentum = 0.5,
        monitoring_batches =  100,
        monitoring_dataset =  trainset,
        cost = SMD(corruptor=GaussianCorruptor(stdev=0.4)),
        termination_criterion =  EpochCounter(50),
        update_callbacks = ExponentialDecay(decay_factor=1.0001, min_lr=0.0001)
        )
    model = layer
    return Train(model = model, algorithm = train_algo, dataset = trainset, \
            extensions=[MomentumAdjustor(final_momentum=0.9, start=0, saturate=20), ])

def get_hybtrainer(model, trainset, validset=None):
    monitoring_batches = None if validset is None else 100
    train_algo = SGD(
        batch_size = 100,
        init_momentum = 0.5,
        learning_rate = 0.5,
        monitoring_batches = monitoring_batches,
        monitoring_dataset = validset,
        cost = MethodCost(method='cost_from_X', supervised=1),
        termination_criterion = EpochCounter(100),
        update_callbacks = ExponentialDecay(decay_factor=1.0001, min_lr=0.001)
    )
    return Train(model=model, algorithm=train_algo, dataset=trainset, save_freq=0, save_path='epoch', \
            extensions=[MomentumAdjustor(final_momentum=0.95, start=0, saturate=80), ])

def get_output(model, tdata, batch_size=100, squeeze=False):
    # get output submodel classifiers
    Xb = model.get_input_space().make_theano_batch()
    Yb = model.fprop(Xb)
    
    data = tdata.get_topological_view()
    # fill up with zeroes for dividible by batch number
    extralength = batch_size - data.shape[0]%batch_size
    
    if extralength < 100:
        data = np.append(data,np.zeros([extralength, data.shape[1],data.shape[2],data.shape[3]]), axis=0)
        data = data.astype('float32')
    if squeeze:
        data = data.squeeze()
    
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
    
    trainset, validset, testset = [], [], []
    for ii, which_data in enumerate(('melspectrum','specfeat')):
        trset,vaset,teset = Whales.whaledata.get_dataset(which_data, tot=submission)
        trainset.append(trset)
        validset.append(vaset)
        testset.append(teset)
    
    # build layers
    layers = []
    structure = [[67*34, 400], [400, 400], [400, 100], [100, 2]]
    layers.append(get_grbm(structure[0]))
    layers.append(get_rbm(structure[1]))
    layers.append(get_rbm(structure[2]))
    layers.append(get_classifier(structure[3][0]))
    
    dset = 0
    #construct training sets for different layers
    dbn_trainset = [ trainset[dset] ,
                TransformerDataset( raw = trainset[dset], transformer = layers[0] ),
                TransformerDataset( raw = trainset[dset], transformer = StackedBlocks( layers[0:2] )),
                TransformerDataset( raw = trainset[dset], transformer = StackedBlocks( layers[0:3] ))  ]
    dbn_validset = [ validset[dset] ,
                TransformerDataset( raw = validset[dset], transformer = layers[0] ),
                TransformerDataset( raw = validset[dset], transformer = StackedBlocks( layers[0:2] )),
                TransformerDataset( raw = validset[dset], transformer = StackedBlocks( layers[0:3] ))  ]
    
    # construct layer trainers
    layer_trainers = []
    layer_trainers.append(get_rbmtrainer(layers[0], dbn_trainset[0]))
    layer_trainers.append(get_rbmtrainer(layers[1], dbn_trainset[1]))
    layer_trainers.append(get_rbmtrainer(layers[2], dbn_trainset[2]))
    layer_trainers.append(get_hybtrainer(layers[3], dbn_trainset[3], dbn_validset[3]))
    
    #unsupervised pretraining
    for layer_trainer in layer_trainers[0:3]:
        layer_trainer.main_loop()
    
    #supervised training
    layer_trainers[-1].main_loop()
    
