#!/usr/bin/python2

import os
import numpy as np
from theano import function
from pylearn2.train import Train
from pylearn2.models.maxout import MaxoutConvC01B, Maxout
from pylearn2.models.mlp import MLP, Layer, Softmax
from pylearn2.space import Conv2DSpace
from pylearn2.training_algorithms.learning_rule import Momentum, MomentumAdjustor
from pylearn2.training_algorithms.sgd import SGD, ExponentialDecay, LinearDecayOverEpoch
from pylearn2.termination_criteria import EpochCounter
from pylearn2.costs.mlp.dropout import Dropout
from pylearn2.monitor import Monitor
from pylearn2.utils import serial

import GalaxyZoo.gzdeepdata

DATA_DIR = '/home/nico/Data/GalaxyZoo/'

SUBMODEL = 1

if SUBMODEL == 1:
    nclass = 159
elif SUBMODEL == 2:
    nclass = 8

bsize = 50

import theano.tensor as T
from pylearn2.utils import wraps

class SoftmaxRMSE(Softmax):
    def __init__(self, *args, **kwargs):
        super(SoftmaxRMSE, self).__init__(*args, **kwargs)

    @wraps(Layer.cost)
    def cost(self, Y, Y_hat):
        return T.sqrt(T.sqr(Y - Y_hat).sum(axis=1).mean())


def get_maxout(dim_input):
    config = {
        'batch_size': bsize,
        'input_space': Conv2DSpace(shape=dim_input[:2], num_channels=dim_input[2], axes=['c', 0, 1, 'b']),
        'layers': [
        MaxoutConvC01B(layer_name='h0', num_channels=96, num_pieces=2, irange=.005, tied_b=1, max_kernel_norm=.9,
                       kernel_shape=[8, 8], pool_shape=[4, 4], pool_stride=[2, 2], W_lr_scale=.05, b_lr_scale=.05),
        MaxoutConvC01B(layer_name='h1', num_channels=128, num_pieces=2, irange=.005, tied_b=1, max_kernel_norm=0.9,
                       kernel_shape=[7, 7], pad=3, pool_shape=[4, 4], pool_stride=[2, 2], W_lr_scale=.05, b_lr_scale=.05),
        MaxoutConvC01B(layer_name='h2', num_channels=160, num_pieces=3, irange=.005, tied_b=1, max_kernel_norm=0.9,
                       kernel_shape=[6, 6], pad=2, pool_shape=[2, 2], pool_stride=[2, 2], W_lr_scale=.05, b_lr_scale=.05),
        MaxoutConvC01B(layer_name='h3', num_channels=192, num_pieces=4, irange=.005, tied_b=1, max_kernel_norm=0.9,
                       kernel_shape=[5, 5], pad=1, pool_shape=[2, 2], pool_stride=[2, 2], W_lr_scale=.05, b_lr_scale=.05),
        Maxout(layer_name='h4', irange=.005, num_units=500, num_pieces=5, max_col_norm=1.9),
        Softmax(layer_name='y', n_classes=nclass, irange=.005, max_col_norm=1.9)
        ]
    }
    return MLP(**config)


def get_trainer(model, trainset, epochs=50):
    train_algo = SGD(
        batch_size=bsize,
        learning_rate=0.15,
        learning_rule=Momentum(init_momentum=0.5),
        # monitoring_batches=100,
        # monitoring_dataset=trainset,
        cost=Dropout(input_include_probs={'h0': .8}, input_scales={'h0': 1.}),
        termination_criterion=EpochCounter(epochs),
    )
    path = DATA_DIR + 'model' + str(SUBMODEL) + 'saved_maxoutx.pkl'
    return Train(model=model, algorithm=train_algo, dataset=trainset, save_path=path, save_freq=1,
                 extensions=[MomentumAdjustor(final_momentum=0.7, start=0, saturate=int(epochs*0.4)),
                             LinearDecayOverEpoch(start=1, saturate=int(epochs*0.7), decay_factor=.01)])


def get_output(model, tdata, batch_size=bsize / 2):
    # get output submodel classifiers
    Xb = model.get_input_space().make_theano_batch()
    Yb = model.fprop(Xb)

    propagate = function([Xb], Yb)

    data = tdata.get_topological_view()
    # fill up with zeroes for dividible by batch number
    extralength = batch_size - data.shape[3] % batch_size

    output = []
    for ii in range(int(data.shape[3] / batch_size) + 1):
        seldata = data[:, :, :, ii * batch_size:min((ii+1) * batch_size, data.shape[3])]
        if ii == int(data.shape[3] / batch_size) and extralength < batch_size:
            seldata = np.concatenate((seldata, np.zeros([data.shape[0], data.shape[1], data.shape[2], extralength],
                                                        dtype='float32')), axis=3)
        output.append(propagate(seldata))

    output = np.reshape(output, (len(output) * output[0].shape[0], -1))

    if extralength < batch_size:
        # remove the filler
        output = output[:-extralength, :]

    return output


if __name__ == '__main__':

    trainset, testset = GalaxyZoo.gzdeepdata.get_data()

    # build and train classifiers for submodels
    iters = 100 if SUBMODEL == 1 else 50
    if os.path.exists(DATA_DIR + 'model' + str(SUBMODEL) + 'saved_maxoutx.pkl'):
        model = serial.load(DATA_DIR + 'model' + str(SUBMODEL) + 'saved_maxoutx.pkl')
        iters = 1
        # reset monitor, can't be re-used
        model.monitor = Monitor(model)
    else:
        model = get_maxout([64, 64, 3])
    get_trainer(model, trainset, iters).main_loop()

    outtrainset = get_output(model, trainset)
    np.save(DATA_DIR+'feats_maxoutx_train' + str(SUBMODEL) + '.npy', outtrainset)

    outtestset = get_output(model, testset)
    np.save(DATA_DIR+'feats_maxoutx_test' + str(SUBMODEL) + '.npy', outtestset)
