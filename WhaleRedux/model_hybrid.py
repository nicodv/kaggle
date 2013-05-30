#!/usr/bin/python

import numpy as np
import scipy
import pandas as pd
from sklearn import cross_validation, ensemble, metrics, linear_model

DATA_DIR = '/home/nico/datasets/Kaggle/WhaleRedux/'

def load_data():
    targets = np.load(DATA_DIR+'targets.npy')[:,0]
    
    c2train = np.load(DATA_DIR+'conv2out_train.npy')
    c2test = np.load(DATA_DIR+'conv2out_test.npy')
    c1train = np.load(DATA_DIR+'conv1out_train.npy')
    c1test = np.load(DATA_DIR+'conv1out_test.npy')
    
    spectrain = np.load(DATA_DIR+'trainspecfeat.npy')
    spectrain = np.concatenate((np.mean(spectrain, axis=1),
                                np.std(spectrain, axis=1),
                                scipy.stats.skew(spectrain, axis=1),
                                np.mean(np.diff(spectrain, axis=1), axis=1)), axis=1)
    spectest = np.load(DATA_DIR+'testspecfeat.npy')
    spectest = np.concatenate((np.mean(spectest, axis=1),
                               np.std(spectest, axis=1),
                                scipy.stats.skew(spectest, axis=1),
                                np.mean(np.diff(spectest, axis=1), axis=1)), axis=1)
    
    traindata = np.concatenate((c2train, c1train), axis=1)
    testdata = np.concatenate((c2test, c1test), axis=1)
    
    return traindata, testdata, targets

def train_model(traindata, targets):
    
    models = [ensemble.GradientBoostingClassifier(n_estimators=500, learning_rate=0.05, \
                max_depth=20, subsample=0.5, max_features=80, min_samples_leaf=20)]
    
    # use StratifiedKFold, because survived 0/1 is not evenly distributed
    cv = cross_validation.StratifiedKFold(targets, n_folds=5)
    
    scores = [0]*len(models)
    for i in range(len(models)):
        # get scores
        scores[i] = cross_validation.cross_val_score(models[i], traindata, targets, \
                    cv=cv, n_jobs=-1, score_func=metrics.auc_score)
        print "Cross-validation accuracy on the training set for model %d:" % i
        print "%0.3f (+/-%0.03f)" % (scores[i].mean(), scores[i].std() / 2)
        
        models[i].fit(traindata, targets)
    
    return models

if __name__ == '__main__':
    
    traindata, testdata, targets = load_data()
    
    models = train_model(traindata, targets)
    
    output = models[0].predict_proba(testdata)
    np.savetxt(DATA_DIR+'model_hybrid.csv', output[:,0], delimiter=",")
    
