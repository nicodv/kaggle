#!/usr/bin/python

import os
import numpy as np
from theano import function

from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix, DefaultViewConverter
from pylearn2.datasets import preprocessing
import pylearn2.utils.serial as serial
from pylearn2.train import Train
from pylearn2.models.mlp import MLP, ConvRectifiedLinear, Softmax, Sigmoid, RectifiedLinear
from pylearn2.space import Conv2DSpace
from pylearn2.corruption import GaussianCorruptor
from pylearn2.costs.ebm_estimation import SMD
from pylearn2.models.rbm import RBM, GaussianBinaryRBM
from pylearn2.energy_functions.rbm_energy import GRBM_Type_1
from pylearn2.training_algorithms.sgd import SGD, ExponentialDecay, MomentumAdjustor
from pylearn2.termination_criteria import EpochCounter
from pylearn2.costs.cost import MethodCost
from pylearn2.costs.supervised_cost import CrossEntropy
from pylearn2.base import StackedBlocks
from pylearn2.datasets.transformer_dataset import TransformerDataset
from sklearn.metrics.metrics import auc_score

DATA_DIR = '/home/nico/Code/datasets/Kaggle/Whales/'


class Whales(DenseDesignMatrix):
    
    def __init__(self, which_set, which_data, start=None, stop=None, preprocessor=None):
        assert which_set in ['train','valid','test']
        assert which_data in ['spectrum','melspectrum','specfeat']
        
        t_set = 'train' if which_set=='valid' else which_set
        X = np.load(os.path.join(DATA_DIR,t_set+which_data+'.npy'))
        X = np.cast['float32'](X)
        X = np.reshape(X,(X.shape[0], np.prod(X.shape[1:])))
        
        if which_set == 'test':
            y = np.zeros((X.shape[0],2))
        else:
            y = np.load(os.path.join(DATA_DIR,'targets.npy'))
            
        if start is not None:
            assert start >= 0
            assert stop > start
            assert stop <= X.shape[0]
            X = X[start:stop, :]
            y = y[start:stop]
            assert X.shape[0] == y.shape[0]
            
        if which_data == 'spectrum':
            view_converter = DefaultViewConverter((67,34,1))
        elif which_data == 'melspectrum':
            view_converter = DefaultViewConverter((67,40,1))
        elif which_data == 'specfeat':
            view_converter = DefaultViewConverter((67,1,24))
            
        super(Whales,self).__init__(X=X, y=y, view_converter=view_converter)
        
        assert not np.any(np.isnan(self.X))
        
        if preprocessor:
            preprocessor.apply(self)


def get_dataset(which_data, tot=False):
    train_path = DATA_DIR+'/train'+which_data+'_preprocessed.pkl'
    valid_path = DATA_DIR+'/valid'+which_data+'_preprocessed.pkl'
    tottrain_path = DATA_DIR+'/tottrain'+which_data+'_preprocessed.pkl'
    test_path = DATA_DIR+'/test'+which_data+'_preprocessed.pkl'
    
    if os.path.exists(train_path) and os.path.exists(valid_path) and os.path.exists(test_path):
        
        print 'loading preprocessed data'
        trainset = serial.load(train_path)
        validset = serial.load(valid_path)
        if tot:
            tottrainset = serial.load(tottrain_path)
        testset = serial.load(test_path)
    else:
        
        print 'loading raw data...'
        trainset = Whales(which_set="train", which_data=which_data, start=0, stop=26671)
        validset = Whales(which_set="train", which_data=which_data, start=26671, stop=36671)
        tottrainset = Whales(which_set="train", which_data=which_data)
        testset = Whales(which_set="test", which_data=which_data)
        
        print 'preprocessing data...'
        pipeline = preprocessing.Pipeline()
        
        if which_data in ('spectrum', 'melspectrum'):
            pipeline.items.append(preprocessing.Standardize(global_mean=True, global_std=True))
        else:
            # global_mean/std=False voor per-feature standardization
            pipeline.items.append(preprocessing.Standardize(global_mean=False, global_std=False))
            
        # ZCA = zero-phase component analysis
        # very similar to PCA, but preserves the look of the original image better
        pipeline.items.append(preprocessing.ZCA())
        
        trainset.apply_preprocessor(preprocessor=pipeline, can_fit=True)
        # this uses numpy format for storage instead of pickle, for memory reasons
        trainset.use_design_loc(DATA_DIR+'/train_'+which_data+'_design.npy')
        validset.apply_preprocessor(preprocessor=pipeline, can_fit=False)
        validset.use_design_loc(DATA_DIR+'/valid_'+which_data+'_design.npy')
        tottrainset.apply_preprocessor(preprocessor=pipeline, can_fit=True)
        tottrainset.use_design_loc(DATA_DIR+'/tottrain_'+which_data+'_design.npy')
        # note the can_fit=False: no sharing between train and test data
        testset.apply_preprocessor(preprocessor=pipeline, can_fit=False)
        testset.use_design_loc(DATA_DIR+'/test_'+which_data+'_design.npy')
        
        # this path can be used for visualizing weights after training is done
        trainset.yaml_src = '!pkl: "%s"' % train_path
        validset.yaml_src = '!pkl: "%s"' % valid_path
        tottrainset.yaml_src = '!pkl: "%s"' % tottrain_path
        testset.yaml_src = '!pkl: "%s"' % test_path
        
        print 'saving preprocessed data...'
        serial.save(DATA_DIR+'/train'+which_data+'_preprocessed.pkl', trainset)
        serial.save(DATA_DIR+'/valid'+which_data+'_preprocessed.pkl', validset)
        serial.save(DATA_DIR+'/tottrain'+which_data+'_preprocessed.pkl', tottrainset)
        serial.save(DATA_DIR+'/test'+which_data+'_preprocessed.pkl', testset)
        
    if tot:
        return tottrainset, None, testset
    else:
        return trainset, validset, testset

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

def get_conv2D(dim_input):
    config = {
        'batch_size': 100,
        'input_space': Conv2DSpace(shape=dim_input[:2], num_channels=dim_input[2]),
        'dropout_include_probs': [1, 1, 0.5, 1],
        'dropout_input_include_prob': 0.8,
        'layers': [
        ConvRectifiedLinear(layer_name='h0', output_channels=48, irange=.05, init_bias=-1.,
            kernel_shape=[7, 7], pool_shape=[4, 4], pool_stride=[3, 2]),
        ConvRectifiedLinear(layer_name='h1', output_channels=48, irange=.05, init_bias=-1.,
            kernel_shape=[5, 5], pool_shape=[2, 2], pool_stride=[1, 1]),
        ConvRectifiedLinear(layer_name='h2', output_channels=24, irange=.05, init_bias=-1.,
            kernel_shape=[3, 3], pool_shape=[2, 2], pool_stride=[2, 2]),
        Softmax(layer_name='y', n_classes=2, istdev=.025, W_lr_scale=0.25)
        ]
    }
    return MLP(**config)

def get_conv1D(dim_input):
    config = {
        'batch_size': 100,
        'input_space': Conv2DSpace(shape=dim_input[:2], num_channels=dim_input[2]),
        'dropout_include_probs': [1, 0.5, 1],
        'dropout_input_include_prob': 0.8,
        'layers': [
        ConvRectifiedLinear(layer_name='h0', output_channels=48, irange=.05, init_bias=-1.,
            kernel_shape=[7, 1], pool_shape=[4, 1], pool_stride=[3, 1]),
        ConvRectifiedLinear(layer_name='h1', output_channels=48, irange=.05, init_bias=-1.,
            kernel_shape=[5, 1], pool_shape=[2, 1], pool_stride=[1, 1]),
        ConvRectifiedLinear(layer_name='h2', output_channels=24, irange=.05, init_bias=-1.,
            kernel_shape=[3, 1], pool_shape=[2, 1], pool_stride=[2, 1]),
        Softmax(layer_name='y', n_classes=2, istdev=.025, W_lr_scale=0.25)
        ]
    }
    return MLP(**config)

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

def get_trainer(model, trainset, validset):
    monitoring_batches = None if validset is None else 100
    train_algo = SGD(
        batch_size = 100,
        init_momentum = 0.5,
        learning_rate = 0.5,
        monitoring_batches = monitoring_batches,
        monitoring_dataset = validset,
        cost = MethodCost(method='cost_from_X', supervised=1),
        termination_criterion = EpochCounter(50),
        update_callbacks = ExponentialDecay(decay_factor=1.001, min_lr=0.001)
    )
    return Train(model=model, algorithm=train_algo, dataset=trainset, save_freq=0, save_path='epoch', \
            extensions=[MomentumAdjustor(final_momentum=0.95, start=0, saturate=40), ])

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

def get_logtrainer(model, trainset, validset=None):
    monitoring_batches = None if validset is None else 100
    train_algo = SGD(
        batch_size = 100,
        init_momentum = 0.5,
        learning_rate = 0.5,
        monitoring_batches = monitoring_batches,
        monitoring_dataset = validset,
        cost = CrossEntropy(),
        termination_criterion = EpochCounter(100),
        update_callbacks = ExponentialDecay(decay_factor=1.0001, min_lr=0.001)
    )
    return Train(model=model, algorithm=train_algo, dataset=trainset, save_freq=0, save_path='epoch', \
            extensions=[MomentumAdjustor(final_momentum=0.95, start=0, saturate=40), ])

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
    for ii, which_data in enumerate(('spectrum','melspectrum','specfeat')):
        trset,vaset,teset = get_dataset(which_data, tot=submission)
        trainset.append(trset)
        validset.append(vaset)
        testset.append(teset)
    
    # define submodels
    sizes = [[67,34,1],
             [67,40,1],
             [67,1,24]
            ]
    # build and train classifiers for submodels
    submodels = [get_conv2D(sizes[0]),
                 get_conv2D(sizes[1]),
                 get_conv1D(sizes[2])]
    for ii, which_model in enumerate(submodels):
        get_trainer(submodels[ii], trainset[ii], validset[ii]).main_loop()
    
    # validate submodels
    output = []
    AUC = []
    if not submission:
        for ii, which_model in enumerate(submodels):
            output.append(get_output(which_model,validset[ii]))
            # calculate AUC using sklearn
            AUC.append(auc_score(validset[ii].get_targets()[:,0],output[ii][:,0]))
    else:
        for ii, which_model in enumerate(submodels):
            output.append(get_output(which_model,testset[ii]))
            np.savetxt('/home/nico/Code/datasets/Kaggle/Whales/model'+str(ii)+'.csv', output[ii], delimiter=",")
    
#==============================================================================
#     HYBRID MODEL
#==============================================================================
    # construct data sets for hybrid model
    hybtrainset, hybvalidset, hybtestset = [], [], []
    for ii, which_model in enumerate(submodels):
        del which_model.layers[-1]
        del which_model.dropout_include_probs[-1]
        del which_model.dropout_scales[-1]
        hybtrainset.append(get_output(which_model,trainset[ii]))
        hybvalidset.append(get_output(which_model,validset[ii]))
        hybtestset.append(get_output(which_model,testset[ii]))
    
    # reshape
    train_no = hybtrainset[0].shape[0]
    valid_no = hybvalidset[0].shape[0]
    test_no = hybtestset[0].shape[0]
    hybtrainset = np.swapaxes(np.reshape(hybtrainset,[3,train_no,hybtrainset[0].shape[1]]),0,1)
    hybtrainset = np.reshape(hybtrainset,[train_no,-1])
    hybvalidset = np.swapaxes(np.reshape(hybvalidset,[3,valid_no,hybvalidset[0].shape[1]]),0,1)
    hybvalidset = np.reshape(hybvalidset,[valid_no,-1])
    hybtestset = np.swapaxes(np.reshape(hybtestset,[3,test_no,hybtestset[0].shape[1]]),0,1)
    hybtestset = np.reshape(hybtestset,[test_no,-1])
    # to pylearn2 format
    hybtraindata = DenseDesignMatrix(X=hybtrainset, y=trainset[0].get_targets(), view_converter=DefaultViewConverter((hybtrainset.shape[1])))
    hybvaliddata= DenseDesignMatrix(X=hybvalidset, y=validset[0].get_targets(), view_converter=DefaultViewConverter((hybvalidset.shape[1])))
    hybtestdata= DenseDesignMatrix(X=hybtestset, y=testset[0].get_targets(), view_converter=DefaultViewConverter((hybtestset.shape[1])))
    
    hybmodel = get_classifier(hybtrainset.shape[1])
    get_hybtrainer(hybmodel, hybtraindata, vdata=hybvaliddata).main_loop()
    
    # test hybrid model
    if not submission:
        houtput = get_output(hybmodel, hybvaliddata)
        hAUC = auc_score(validset[0].get_targets()[:,0],houtput[:,0])
    else:
        # Right Whale, Holy Grail!
        houtput = get_output(hybmodel, hybtestdata)
        np.savetxt('/home/nico/Code/Kaggle/Whales/hybmodel.csv', houtput, delimiter=",")
    
#==============================================================================
#     DEEP BELIEF NETWORK
#==============================================================================
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
    
