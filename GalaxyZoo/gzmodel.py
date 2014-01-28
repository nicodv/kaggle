#!/usr/bin/python2

import numpy as np
from theano import function

from pylearn2.train import Train
from pylearn2.models.mlp import MLP, ConvRectifiedLinear, Softmax, Linear, Sigmoid, RectifiedLinear
from pylearn2.space import Conv2DSpace
from pylearn2.training_algorithms.sgd import SGD, ExponentialDecay, MomentumAdjustor
from pylearn2.termination_criteria import EpochCounter
from pylearn2.costs.cost import MethodCost
from sklearn.metrics.metrics import auc_score

import GalaxyZoo

DATA_DIR = '/home/nico/Data/GalaxyZoo/'


def get_conv(dim_input):
    config = {
        'batch_size': 200,
        'input_space': Conv2DSpace(shape=dim_input[:2], num_channels=dim_input[2]),
        'dropout_include_probs': [1, 1, 1, 0.5, 1],
        'dropout_input_include_prob': 0.8,
        'layers': [
        ConvRectifiedLinear(layer_name='h0', output_channels=40, irange=.04, init_bias=0.5, max_kernel_norm=1.9365,
            kernel_shape=[7, 7], pool_shape=[6, 4], pool_stride=[3, 2], W_lr_scale=0.64),
        ConvRectifiedLinear(layer_name='h1', output_channels=30, irange=.05, init_bias=0., max_kernel_norm=1.9365,
            kernel_shape=[5, 5], pool_shape=[2, 2], pool_stride=[1, 1], W_lr_scale=1.),
        ConvRectifiedLinear(layer_name='h2', output_channels=20, irange=.05, init_bias=0., max_kernel_norm=1.9365,
            kernel_shape=[5, 5], pool_shape=[4, 4], pool_stride=[1, 1], W_lr_scale=1.),
        ConvRectifiedLinear(layer_name='h3', output_channels=10, irange=.05, init_bias=0., max_kernel_norm=1.9365,
            kernel_shape=[3, 3], pool_shape=[2, 2], pool_stride=[2, 2], W_lr_scale=1.),
        Softmax(layer_name='y', n_classes=2, istdev=.025, W_lr_scale=0.25)
        ]
    }
    return MLP(**config)


def get_trainer(model, trainset, validset, epochs=50):
    monitoring_batches = None if validset is None else 50
    train_algo = SGD(
        batch_size = 200,
        init_momentum = 0.5,
        learning_rate = 0.5,
        monitoring_batches = monitoring_batches,
        monitoring_dataset = validset,
        cost = MethodCost(method='cost_from_X', supervised=1),
        termination_criterion = EpochCounter(epochs),
        update_callbacks = ExponentialDecay(decay_factor=1.0005, min_lr=0.001)
    )
    return Train(model=model, algorithm=train_algo, dataset=trainset, save_freq=0, save_path='epoch',
                 extensions=[MomentumAdjustor(final_momentum=0.95, start=0, saturate=int(epochs*0.8)), ])


def get_output(model, tdata, layerindex, batch_size=200):
    # get output submodel classifiers
    Xb = model.get_input_space().make_theano_batch()
    Yb = model.fprop(Xb, apply_dropout=False, return_all=True)

    data = tdata.get_topological_view()
    # fill up with zeroes for dividible by batch number
    extralength = batch_size - data.shape[0] % batch_size

    if extralength < batch_size:
        data = np.append(data,np.zeros([extralength, data.shape[1],data.shape[2],data.shape[3]]), axis=0)
        data = data.astype('float32')

    propagate = function([Xb], Yb)

    output = []
    for ii in range(int(data.shape[0]/batch_size)):
        seldata = data[ii*batch_size:(ii+1)*batch_size,:]
        output.append(propagate(seldata)[layerindex])

    output = np.reshape(output, (data.shape[0],-1))

    if extralength < batch_size:
        # remove the filler
        output = output[:-extralength]

    return output


if __name__ == '__main__':

    submission = True

    ####################
    # MEL SPECTRUM #
    ####################
    trainset, validset, testset = GalaxyZoo.gzdata.get_dataset('melspectrum', tot=submission)

    # build and train classifiers for submodels
    model = get_conv([424, 424, 1])
    get_trainer(model, trainset, validset, 80).main_loop()

    # validate model
    if not submission:
        output = get_output(model, validset)
        # calculate AUC using sklearn
        AUC = auc_score(validset.get_targets()[:,0],output[:,0])
        print(AUC)
    else:
        outtestset = get_output(model,testset,-1)
        # save test output as submission
        np.savetxt(DATA_DIR+'model_conv2net.csv', outtestset[:,0], delimiter=",")

        # construct data sets with model output
        outtrainset = get_output(model,trainset,-2)
        outtestset = get_output(model,testset,-2)

        np.save(DATA_DIR+'conv2altout_train', outtrainset)
        np.save(DATA_DIR+'conv2altout_test', outtestset)
