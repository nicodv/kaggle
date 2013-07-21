#!/usr/bin/python

import os
import numpy as np
import pandas as pd
from theano import function
import Digits.digits_data

from pylearn2.utils import serial
from pylearn2.train import Train
from pylearn2.models.mlp import MLP, Softmax
from pylearn2.models.maxout import MaxoutConvC01B
from pylearn2.costs.mlp.dropout import Dropout
from pylearn2.space import Conv2DSpace
from pylearn2.training_algorithms.sgd import SGD, ExponentialDecay, MomentumAdjustor
from pylearn2.termination_criteria import EpochCounter, MonitorBased
from sklearn.metrics.metrics import accuracy_score
from sklearn import cross_validation, linear_model

DATA_DIR = '/home/nico/datasets/Kaggle/Digits/'


def get_maxout(dim_input, batch_size=100):
    config = {
        'batch_size': batch_size,
        'input_space': Conv2DSpace(shape=dim_input[:2], num_channels=dim_input[2], axes=['c', 0, 1, 'b']),
        'layers': [
        MaxoutConvC01B(layer_name='h0', pad=0, num_channels=48, num_pieces=2, kernel_shape=[8, 8],
                     pool_shape=[4, 4], pool_stride=[2, 2], irange=.005, max_kernel_norm=.9),
        MaxoutConvC01B(layer_name='h1', pad=3, num_channels=48, num_pieces=2, kernel_shape=[8, 8],
                     pool_shape=[4, 4], pool_stride=[2, 2], irange=.005, max_kernel_norm=1.9365),
        MaxoutConvC01B(layer_name='h2', pad=3, num_channels=24, num_pieces=4, kernel_shape=[5, 5],
                     pool_shape=[2, 2], pool_stride=[2, 2], irange=.005, max_kernel_norm=1.9365),
        Softmax(layer_name='y', max_col_norm=1.9365, n_classes=10, irange=0.005)
        ]
    }
    return MLP(**config)

def get_trainer(model, trainset, validset, epochs=20, batch_size=100):
    monitoring_batches = None if validset is None else 20
    train_algo = SGD(
        batch_size = batch_size,
        init_momentum = 0.5,
        learning_rate = 0.05,
        monitoring_batches = monitoring_batches,
        monitoring_dataset = validset,
        cost = Dropout(input_include_probs={'h0': 0.8},
                        input_scales={'h0': 1.},
                        default_input_include_prob=0.5, default_input_scale=1./0.5),
        #termination_criterion = MonitorBased(channel_name='y_misclass', prop_decrease=0., N=50),
        termination_criterion = EpochCounter(epochs),
        update_callbacks = ExponentialDecay(decay_factor=1.00004, min_lr=0.000001)
    )
    return Train(model=model, algorithm=train_algo, dataset=trainset, save_freq=0, save_path='epoch', \
            extensions=[MomentumAdjustor(final_momentum=0.7, start=0, saturate=int(0.8*epochs))])

def get_output(model, tdata, layerindex, batch_size=100):
    # get output submodel classifiers
    Xb = model.get_input_space().make_theano_batch()
    Yb = model.fprop(Xb, return_all=True)
    
    data = tdata.get_topological_view() #.transpose((3,1,2,0))
    # fill up with zeros to keep dividible by batch number
    extralength = batch_size - data.shape[3]%batch_size
    
    if extralength < batch_size:
        data = np.append(data,np.zeros([extralength, data.shape[0],data.shape[1],data.shape[2]]), axis=3)
        data = data.astype('float32')
    
    propagate = function([Xb], Yb, allow_input_downcast=True)
    
    output = []
    for ii in xrange(int(data.shape[3]/batch_size)):
        seldata = data[:,:,:,ii*batch_size:(ii+1)*batch_size]
        output.append(propagate(seldata)[layerindex])
    
    output = np.reshape(output,[data.shape[3],-1])
    
    if extralength < batch_size:
        # remove the filler
        output = output[:-extralength,:]
    
    return output

def get_comb_models(traindata, targets, crossval=True):
    # traindata: list with NumExamples * NumOutputs(=10) array with length 'no. preprocessors'
    # reshape to NumExamples * [NumPreprocessors * NumOutputs]
    traindata = np.array(traindata).transpose((1,0,2))
    traindata = np.reshape(traindata,[traindata.shape[0],-1])
    
    # needs to be not one-hot
    targets = targets.argmax(axis=1)
    
    models = [linear_model.LogisticRegression(penalty='l1', dual=False, C=5., fit_intercept=False),
                linear_model.LogisticRegression(penalty='l2', dual=False, C=10., fit_intercept=False),
                linear_model.LogisticRegression(penalty='l2', dual=False, C=20., fit_intercept=True)]
    
    if crossval:
        # use StratifiedKFold, because survived 0/1 is not evenly distributed
        cv = cross_validation.StratifiedKFold(targets, n_folds=5)
        scores = [0]*len(models)
    
    for ii in range(len(models)):
        if crossval:
            # get scores
            scores[ii] = cross_validation.cross_val_score(models[ii], traindata, targets, \
                        cv=cv, n_jobs=1, scoring='accuracy')
            print "Cross-validation accuracy on the training set for model %d:" % ii
            print "%0.3f (+/-%0.03f)" % (scores[ii].mean(), scores[ii].std() / 2)
        else:
            models[ii].fit(traindata, targets)
    
    return models

if __name__ == '__main__':
    
    submission = True
    batch_size = 128
    
    # ---------------------------------------------------------------------------
    # METHOD 1, combining submodels
    # ---------------------------------------------------------------------------
#    preprocessors = ('normal', 'hshear', 'vshear', 'rotate', 'noisy', 'patch', 'emboss')
#    epochs = [200, 300, 300, 300, 300, 300, 300]
#    
#    models = [0]*len(preprocessors)
#    accuracies = [0]*len(preprocessors)
#    outtrainset = [0]*len(preprocessors)
#    outvalidset = [0]*len(preprocessors)
#    outtestset = [0]*len(preprocessors)
#    for ii, preprocessor in enumerate(preprocessors):
#        trainset,validset,testset = Digits.digits_data.get_dataset(tot=submission, preprocessor=preprocessor)
#        
#        if not os.path.exists(DATA_DIR+preprocessor+'_model.pkl'):
#            # build and train classifiers for submodels
#            models[ii] = get_maxout([28,28,1], batch_size=batch_size)
#            get_trainer(models[ii], trainset, validset, epochs=epochs[ii], batch_size=batch_size).main_loop()
#        else:
#            models[ii] = serial.load(DATA_DIR+preprocessor+'_model.pkl')
#        
#        outtrainset[ii] = get_output(models[ii],trainset,-1)
#        
#        if not submission:
#            # validset is used to evaluate maxout network performance
#            outvalidset[ii] = get_output(models[ii],validset,-1)
#            accuracies[ii] = accuracy_score(np.argmax(validset.get_targets(),axis=1),np.argmax(outvalidset[ii],axis=1))
#        else:
#            outtestset[ii] = get_output(models[ii],testset,-1)
#            serial.save(DATA_DIR+preprocessor+'_model.pkl', models[ii])
#    
#    if not submission:
#        print(accuracies)
#    
#    # now combine maxout predictions in a bunch of classifiers
#    comboutputs = []
#    if not submission:
#        # outtrainset is used to evaluate comb models
#        cmodels = get_comb_models(outtrainset, trainset.y, crossval=True)
#    else:
#        cmodels = get_comb_models(outtrainset, trainset.y, crossval=False)
#        # outtestset: list with NumExamples * NumOutputs(=10) array with length 'no. preprocessors'
#        # reshape to NumExamples * [NumPreprocessors * NumOutputs]
#        outtestseta = np.array(outtestset).transpose((1,0,2))
#        outtestseta = np.reshape(outtestseta,[outtestseta.shape[0],-1])
#        
#        for ii in range(len(cmodels)):
#            comboutputs.append(cmodels[ii].predict_proba(outtestseta))
#        
#        # take mean of classifiers and save output as submission
#        ImageId = range(1,28001)
#        subm = np.argmax(np.mean(comboutputs, axis=0),axis=1)
#        
#        # or just take the mean...
#        simple = np.mean(outtestseta.reshape([28000,len(preprocessors),10]),axis=1)
#        simplesubm = np.argmax(simple,axis=1)
#        
#        pdsubm = pd.DataFrame({'ImageId': ImageId, 'Label': subm})
#        pdsubm.to_csv(DATA_DIR+'submission.csv', header=True, index=False, fmt='%1.0f')
#    
    # ---------------------------------------------------------------------------
    # METHOD 2, train 1 model with distorted data
    # ---------------------------------------------------------------------------
    preprocessors = ('normal', 'hshear', 'vshear', 'rotate', 'noisy', 'patch', 'emboss')
    
    trainset,validset,testset = Digits.digits_data.get_all_datasets(tot=submission, preprocessors=preprocessors)
    
    # build and train classifiers for submodels
    model = get_maxout([28,28,1], batch_size=batch_size)
    get_trainer(model, trainset, validset, epochs=500, batch_size=batch_size).main_loop()
    
    output = get_output(model,testset,-1)
    
    pdsubm = pd.DataFrame({'ImageId': range(1,28001), 'Label': np.argmax(output, axis=1)})
    pdsubm.to_csv(DATA_DIR+'submission.csv', header=True, index=False, fmt='%1.0f')
