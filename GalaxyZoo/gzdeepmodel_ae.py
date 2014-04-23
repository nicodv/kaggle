#!/usr/bin/python2

import os
import numpy as np
from theano import function
from pylearn2.train import Train
from pylearn2.datasets.transformer_dataset import TransformerDataset
from pylearn2.base import StackedBlocks
from pylearn2.models.mlp import MLP, Layer, ConvRectifiedLinear, Softmax, Linear, Sigmoid, RectifiedLinear
import pylearn2.models.autoencoder as autoencoder
from pylearn2.training_algorithms.learning_rule import Momentum, MomentumAdjustor
from pylearn2.training_algorithms.sgd import SGD, ExponentialDecay, LinearDecayOverEpoch
from pylearn2.termination_criteria import EpochCounter
from pylearn2.costs.mlp.dropout import Dropout
from pylearn2.costs import cost
from pylearn2.costs.autoencoder import MeanBinaryCrossEntropy, MeanSquaredReconstructionError
from pylearn2.corruption import GaussianCorruptor, BinomialCorruptor
from pylearn2.monitor import Monitor
from pylearn2.utils import serial

import GalaxyZoo.gzdeepdata

DATA_DIR = '/home/nico/Data/GalaxyZoo/'

SUBMODEL = 1

if SUBMODEL == 1:
    nclass = 159
elif SUBMODEL == 2:
    nclass = 8

bsize = 120


def construct_ae(structure):
    # some settings
    irange = 0.1

    layers = []
    for vsize, hsize in zip(structure[:-1], structure[1:]):
        # DenoisingAutoencoder / ContractiveAutoencoder / HigherOrderContractiveAutoencoder
        layers.append(autoencoder.DenoisingAutoencoder(
            nvis=vsize,
            nhid=hsize,
            tied_weights=True,
            act_enc='sigmoid',
            act_dec='sigmoid',
            irange=irange,
            # for DenoisingAutoencoder / HigherOrderContractiveAutoencoder:
            corruptor=BinomialCorruptor(0.5),
            # for HigherOrderContractiveAutoencoder:
            # num_corruptions=6
        ))
    return StackedBlocks(layers)


def get_ae_pretrainer(layer, data, batch_size, epochs=30):
    init_lr = 0.05
    
    train_algo = SGD(
        batch_size=batch_size,
        learning_rate=init_lr,
        learning_rule=Momentum(init_momentum=0.5),
        monitoring_batches=batch_size,
        monitoring_dataset=data,
        # for ContractiveAutoencoder:
        # cost=cost.SumOfCosts(costs=[[1., MeanSquaredReconstructionError()],
        #                             [0.5, cost.MethodCost(method='contraction_penalty')]]),
        # for HigherOrderContractiveAutoencoder:
        # cost=cost.SumOfCosts(costs=[[1., MeanSquaredReconstructionError()],
        #                             [0.5, cost.MethodCost(method='contraction_penalty')],
        #                             [0.5, cost.MethodCost(method='higher_order_penalty')]]),
        # for DenoisingAutoencoder:
        cost=MeanSquaredReconstructionError(),
        termination_criterion=EpochCounter(epochs)
        )
    return Train(model=layer, algorithm=train_algo, dataset=data,
                 extensions=[MomentumAdjustor(final_momentum=0.9, start=0, saturate=25),
                             LinearDecayOverEpoch(start=1, saturate=25, decay_factor=.02)])


def construct_dbn_from_stack(stack):
    # some settings
    irange = 0.05
    
    layers = []
    for ii, layer in enumerate(stack.layers()):
        layers.append(Sigmoid(
            dim=layer.nhid,
            layer_name='h'+str(ii),
            irange=irange,
            max_col_norm=2.
        ))
    nc = 159 if SUBMODEL == 1 else 8
    # softmax layer at then end for classification
    layers.append(Softmax(
        n_classes=nc,
        layer_name='y',
        irange=irange
    ))
    dbn = MLP(layers=layers, nvis=stack.layers()[0].get_input_space().dim)
    # copy weigths to DBN
    for ii, layer in enumerate(stack.layers()):
        dbn.layers[ii].set_weights(layer.get_weights())
        dbn.layers[ii].set_biases(layer.hidbias.get_value(borrow=False))
    return dbn


def get_finetuner(model, trainset, batch_size=100, epochs=100):
    train_algo = SGD(
        batch_size=batch_size,
        learning_rule=Momentum(init_momentum=0.5),
        learning_rate=0.5,
        monitoring_batches=batch_size,
        monitoring_dataset=trainset,
        cost=Dropout(input_include_probs={'h0': .5}, input_scales={'h0': 2.}),
        termination_criterion=EpochCounter(epochs)
    )
    path = DATA_DIR+'model' + str(SUBMODEL) + 'saved_daex.pkl'
    return Train(model=model, algorithm=train_algo, dataset=trainset, save_path=path, save_freq=10,
                 extensions=[MomentumAdjustor(final_momentum=0.9, start=0, saturate=int(epochs*0.8)),
                             LinearDecayOverEpoch(start=1, saturate=int(epochs*0.7), decay_factor=.02)])


def get_output(model, tdata, batch_size=bsize / 2):
    # get output submodel classifiers
    Xb = model.get_input_space().make_theano_batch()
    Yb = model.fprop(Xb)

    propagate = function([Xb], Yb)

    data = tdata.X
    # fill up with zeroes for dividible by batch number
    extralength = batch_size - data.shape[0] % batch_size

    output = []
    for ii in range(int(data.shape[0] / batch_size) + 1):
        seldata = data[ii * batch_size:min((ii+1) * batch_size, data.shape[0]), :]
        if ii == int(data.shape[0] / batch_size) and extralength < batch_size:
            seldata = np.concatenate((seldata, np.zeros([extralength, data.shape[1]],
                                                        dtype='float32')), axis=0)
        output.append(propagate(seldata))

    output = np.reshape(output, (len(output) * output[0].shape[0], -1))

    if extralength < batch_size:
        # remove the filler
        output = output[:-extralength, :]

    return output


if __name__ == '__main__':

    trainset, testset = GalaxyZoo.gzdeepdata.get_data(flatgrey=True)

    structure = [4096, 3000, 2000, 2000, 2000, 2000]

    # pre-train model
    # stack = construct_ae(structure)
    # for ii, layer in enumerate(stack.layers()):
    #     utraindata = trainset if ii == 0 else TransformerDataset(raw=trainset,
    #                                                              transformer=StackedBlocks(stack.layers()[:ii]))
    #     pretrainer = get_ae_pretrainer(layer, utraindata, bsize, epochs=30)
    #     pretrainer.main_loop()
    # serial.save(DATA_DIR+'daex_pretrained.pkl', stack)
    stack = serial.load(DATA_DIR + 'daex_pretrained.pkl')

    # construct DBN
    dbn = construct_dbn_from_stack(stack)

    # finetune softmax layer a bit
    finetuner = get_finetuner(dbn, trainset, bsize, epochs=15)
    finetuner.main_loop()

    # now finetune layer-by-layer, boost earlier layers
    lrs = [8., 6., 4., 2., 1.]
    for ii, lr in zip(range(len(structure)-1), lrs):
        dbn.monitor = Monitor(dbn)
        # set lr to boosted value for current layer
        dbn.layers[ii].W_lr_scale = lr

        finetuner = get_finetuner(dbn, trainset, bsize, epochs=30)
        finetuner.main_loop()

        # return to original lr
        dbn.layers[ii].W_lr_scale = 1.

    # total finetuner
    # dbn = serial.load(DATA_DIR + 'model' + str(SUBMODEL) + 'saved_daex.pkl')
    dbn.monitor = Monitor(dbn)
    finetuner = get_finetuner(dbn, trainset, bsize, epochs=150)
    finetuner.main_loop()

    outtrainset = get_output(dbn, trainset)
    np.save(DATA_DIR+'feats_daex_train' + str(SUBMODEL) + '.npy', outtrainset)

    outtestset = get_output(dbn, testset)
    np.save(DATA_DIR+'feats_daex_test' + str(SUBMODEL) + '.npy', outtestset)
