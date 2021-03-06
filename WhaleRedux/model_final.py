#!/usr/bin/python

import os
import numpy as np
import mdp
import Oger
import pandas as pd
from sklearn import cross_validation, ensemble, decomposition

DATA_DIR = '/home/nico/datasets/Kaggle/WhaleRedux/'

def load_data():
    targets = np.load(DATA_DIR+'targets.npy')[:,0]
    
    c2train = np.load(DATA_DIR+'conv2out_train.npy')
    c2test = np.load(DATA_DIR+'conv2out_test.npy')
    c1train = np.load(DATA_DIR+'conv1out_train.npy')
    c1test = np.load(DATA_DIR+'conv1out_test.npy')
    
    traindata = np.concatenate((c2train, c1train), axis=1)
    testdata = np.concatenate((c2test, c1test), axis=1)
    
    norm = mdp.nodes.StandardScalerScikitsLearnNode()
    norm.train(traindata)
    traindata = norm.execute(traindata)
    testdata = norm.execute(testdata)
    
    return traindata, testdata, targets

def run_reservoir(data):
    reservoir = Oger.nodes.LeakyReservoirNode(leak_rate=0.4, input_dim=480, output_dim=2000, \
        spectral_radius=0.9, bias_scaling=0, input_scaling=0.2)
    return reservoir.execute(data)

def get_preprocessor(data):
    preprocessor = decomposition.PCA(n_components=500)
    preprocessor.fit(data)
    return preprocessor

def get_classifier(traindata, targets):
    
    models = [ensemble.GradientBoostingClassifier(n_estimators=400, learning_rate=0.05, \
                max_depth=60, subsample=0.5, max_features=120, min_samples_leaf=20)]
    
    # use StratifiedKFold, because survived 0/1 is not evenly distributed
    cv = cross_validation.StratifiedKFold(targets, n_folds=5)
    
    scores = [0]*len(models)
    for i in range(len(models)):
        # get scores
        scores[i] = cross_validation.cross_val_score(models[i], traindata, targets, \
                    cv=cv, n_jobs=-1, scoring='roc_auc')
        print "Cross-validation accuracy on the training set for model %d:" % i
        print "%0.3f (+/-%0.03f)" % (scores[i].mean(), scores[i].std() / 2)
        
        models[i].fit(traindata, targets)
    
    return models

if __name__ == '__main__':
    
    traindata, testdata, targets = load_data()
    fn = np.load(os.path.join(DATA_DIR,'filenames.npy'))
    
    # run reservoir
    res_train = run_reservoir(traindata)
    res_test = run_reservoir(testdata)
    
    # combine data
    tottraindata = np.concatenate((traindata, res_train), axis=1)
    tottestdata = np.concatenate((testdata, res_test), axis=1)
    
    # preprocess data
    preproc = get_preprocessor(tottraindata)
    proc_train = preproc.transform(tottraindata)
    proc_test = preproc.transform(tottestdata)
    
    # now define and train model
    models = get_classifier(proc_train, targets)
    
    output = models[0].predict_proba(proc_test)
    
    # save test output as submission
    subm = pd.DataFrame({'clip': fn, 'probability': output[:,0]})
    subm.to_csv(DATA_DIR+'model_hybrid.csv', header=True, index=False)
