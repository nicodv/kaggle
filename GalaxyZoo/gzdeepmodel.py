#!/usr/bin/python2

import math
import numpy as np
from theano import function
from pylearn2.train import Train
from pylearn2.models.maxout import MaxoutConvC01B, Maxout
from pylearn2.models.mlp import MLP, Layer, ConvRectifiedLinear, Softmax, Linear, Sigmoid, RectifiedLinear
from pylearn2.space import Conv2DSpace
from pylearn2.training_algorithms.learning_rule import Momentum, MomentumAdjustor
from pylearn2.training_algorithms.sgd import SGD, ExponentialDecay
from pylearn2.termination_criteria import EpochCounter
from pylearn2.costs.mlp.dropout import Dropout
from sklearn.metrics import mean_squared_error

import GalaxyZoo.gzdeepdata

DATA_DIR = '/home/nico/Data/GalaxyZoo/'

SUBMODEL = 2

if SUBMODEL == 1:
    nclass = 159
elif SUBMODEL == 2:
    nclass = 8

bsize = 60

import theano.tensor as T
from pylearn2.utils import wraps

class SoftmaxRMSE(Softmax):
    def __init__(self, *args, **kwargs):
        super(SoftmaxRMSE, self).__init__(*args, **kwargs)

    @wraps(Layer.cost)
    def cost(self, Y, Y_hat):
        return T.sqrt(T.sqr(Y - Y_hat).sum(axis=1).mean())


def get_conv1(dim_input):
    config = {
        'batch_size': bsize,
        'input_space': Conv2DSpace(shape=dim_input[:2], num_channels=dim_input[2], axes=['c', 0, 1, 'b']),
        'layers': [
        MaxoutConvC01B(layer_name='h0', num_channels=32, num_pieces=2, irange=.01, init_bias=0., max_kernel_norm=0.9,
                       kernel_shape=[7, 7], pool_shape=[2, 2], pool_stride=[2, 2], W_lr_scale=.1, b_lr_scale=.1),
        MaxoutConvC01B(layer_name='h1', num_channels=48, num_pieces=2, irange=.01, init_bias=0., max_kernel_norm=0.9,
                       kernel_shape=[5, 5], pool_shape=[4, 4], pool_stride=[4, 4], W_lr_scale=.1, b_lr_scale=.1),
        MaxoutConvC01B(layer_name='h2', num_channels=64, num_pieces=2, irange=.01, init_bias=0., max_kernel_norm=0.9,
                       kernel_shape=[5, 5], pool_shape=[2, 2], pool_stride=[2, 2], W_lr_scale=.1, b_lr_scale=.1),
        Maxout(layer_name='h3', irange=0.01, num_units=800, num_pieces=4, max_col_norm=0.9),
        # RectifiedLinear(layer_name='h2', dim=200, sparse_init=15),
        # Sigmoid(layer_name='h2', dim=200, istdev=.005),
        SoftmaxRMSE(layer_name='y', n_classes=nclass, istdev=.01, W_lr_scale=1.)
        ]
    }
    return MLP(**config)


def get_conv2(dim_input):
    config = {
        'batch_size': bsize,
        'input_space': Conv2DSpace(shape=dim_input[:2], num_channels=dim_input[2], axes=['c', 0, 1, 'b']),
        'layers': [
        MaxoutConvC01B(layer_name='h0', num_channels=32, num_pieces=2, irange=.01, init_bias=0., max_kernel_norm=0.9,
                       kernel_shape=[7, 7], pool_shape=[2, 2], pool_stride=[2, 2], W_lr_scale=.2, b_lr_scale=.2),
        MaxoutConvC01B(layer_name='h1', num_channels=48, num_pieces=2, irange=.01, init_bias=0., max_kernel_norm=0.9,
                       kernel_shape=[5, 5], pool_shape=[2, 2], pool_stride=[2, 2], W_lr_scale=.2, b_lr_scale=.2),
        MaxoutConvC01B(layer_name='h2', num_channels=64, num_pieces=4, irange=.01, init_bias=0., max_kernel_norm=0.9,
                       kernel_shape=[5, 5], pool_shape=[2, 2], pool_stride=[2, 2], W_lr_scale=.2, b_lr_scale=.2),
        MaxoutConvC01B(layer_name='h3', num_channels=64, num_pieces=4, irange=.01, init_bias=0., max_kernel_norm=0.9,
                       kernel_shape=[3, 3], pool_shape=[2, 2], pool_stride=[2, 2], W_lr_scale=.2, b_lr_scale=.2),
        Maxout(layer_name='h4', irange=0.01, num_units=400, num_pieces=3, max_col_norm=0.9),
        # RectifiedLinear(layer_name='h3', dim=400, sparse_init=15, max_col_norm=0.9),
        # Sigmoid(layer_name='h2', dim=200, istdev=.005),
        SoftmaxRMSE(layer_name='y', n_classes=nclass, istdev=.01, W_lr_scale=1.)
        ]
    }
    return MLP(**config)


def get_trainer1(model, trainset, validset, epochs=50):
    monitoring_batches = None if validset is None else bsize
    train_algo = SGD(
        batch_size=bsize,
        learning_rate=0.1,
        learning_rule=Momentum(init_momentum=0.5),
        monitoring_batches=monitoring_batches,
        monitoring_dataset=trainset,
        cost=Dropout(input_include_probs={'h0': .8}, input_scales={'h0': 1.}),
        termination_criterion=EpochCounter(epochs),
        update_callbacks=ExponentialDecay(decay_factor=1.00005, min_lr=0.00001)
    )
    return Train(model=model, algorithm=train_algo, dataset=trainset,
                 extensions=[MomentumAdjustor(final_momentum=0.95, start=0, saturate=int(epochs*0.8)), ])


def get_trainer2(model, trainset, validset, epochs=50):
    monitoring_batches = None if validset is None else bsize
    train_algo = SGD(
        batch_size=bsize,
        learning_rate=0.1,
        learning_rule=Momentum(init_momentum=0.5),
        monitoring_batches=monitoring_batches,
        monitoring_dataset=trainset,
        cost=Dropout(input_include_probs={'h0': .8}, input_scales={'h0': 1.}),
        termination_criterion=EpochCounter(epochs),
        update_callbacks=ExponentialDecay(decay_factor=1.00005, min_lr=0.00001)
    )
    return Train(model=model, algorithm=train_algo, dataset=trainset,
                 extensions=[MomentumAdjustor(final_momentum=0.95, start=0, saturate=int(epochs*0.8)), ])


def get_output(model, tdata, layerindex, batch_size=bsize / 2):
    # get output submodel classifiers
    Xb = model.get_input_space().make_theano_batch()
    Yb = model.fprop(Xb, return_all=True)

    data = tdata.get_topological_view()
    # fill up with zeroes for dividible by batch number
    extralength = batch_size - data.shape[3] % batch_size

    propagate = function([Xb], Yb)

    output = []
    for ii in range(int(data.shape[3] / batch_size)):
        seldata = data[:, :, :, ii * batch_size:min((ii+1) * batch_size, data.shape[3])]
        if ii and extralength < batch_size:
            seldata = np.concatenate((seldata, np.zeros([data.shape[0], data.shape[1], data.shape[2], extralength], dtype='float32')), axis=3)
        output.append(propagate(seldata)[layerindex])

    output = np.reshape(output, (data.shape[3], -1))

    if extralength < batch_size:
        # remove the filler
        output = output[:-extralength, :]

    return output


if __name__ == '__main__':

    submission = True

    trainset, validset, testset = GalaxyZoo.gzdeepdata.get_data(tot=submission)

    # build and train classifiers for submodels
    if SUBMODEL == 1:
        model = get_conv1([64, 64, 3])
        get_trainer1(model, trainset, validset, 1).main_loop()
    elif SUBMODEL == 2:
        model = get_conv2([64, 64, 3])
        get_trainer2(model, trainset, validset, 1).main_loop()

    # validate model
    if not submission:
        output = get_output(model, validset, -1)

        # calculate RMSE on original targets using sklearn
        RMSE = math.sqrt(mean_squared_error(validset.get_targets(), output))
        print(RMSE)
    else:
        outtrainset = get_output(model, trainset, -1)
        np.save(DATA_DIR+'feats_convnet_train' + str(SUBMODEL) + '.npy', outtrainset)

        outtestset = get_output(model, testset, -1)
        np.save(DATA_DIR+'feats_convnet_test' + str(SUBMODEL) + '.npy', outtestset)
