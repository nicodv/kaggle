#!/usr/bin/python

import numpy as np
import pandas as pd
from sklearn import cross_validation, ensemble, metrics, linear_model

DATA_DIR = '/home/nico/datasets/Kaggle/GenderWrite/'

def load_data():
    mtrain = np.load(DATA_DIR+'maxout_train.npy')
    mtest = np.load(DATA_DIR+'maxout_test.npy')
    
    feattrain = np.load(DATA_DIR+'feat_train.npy')
    feattest = np.load(DATA_DIR+'feat_test.npy')
    
    traindata = np.concatenate((mtrain, feattrain), axis=1)
    testdata = np.concatenate((mtest, feattest), axis=1)
    
    targets_pp = np.load(DATA_DIR+'targets_per_page.npy')[:,0]
    targets = np.load(DATA_DIR+'targets.npy')[:,0]
    
    return traindata, testdata, targets_pp, targets

def train_comb_model(traindata, targets):
    
    models = [
                ensemble.GradientBoostingClassifier(n_estimators=500, learning_rate=0.05, \
                max_depth=20, subsample=0.5, max_features=80, min_samples_leaf=20)
#                linear_model.LogisticRegression(penalty='l2', dual=True, C=1),
#                ensemble.RandomForestClassifier(n_estimators=500, max_features='log2', \
#                compute_importances=False, oob_score=False, min_samples_leaf=20, criterion='entropy')
                ]
    
    cv = cross_validation.KFold(targets, n_folds=5)
    
    scores = [0]*len(models)
    for i in range(len(models)):
        # get scores
        scores[i] = cross_validation.cross_val_score(models[i], traindata, targets, \
                    cv=cv, n_jobs=-1, score_func=metrics.accuracy_score)
        print "Cross-validation accuracy on the training set for model %d:" % i
        print "%0.3f (+/-%0.03f)" % (scores[i].mean(), scores[i].std() / 2)
        
        models[i].fit(traindata, targets)
    
    return models

def_train_final_model(traindata, targets):
    
    model = linear_model.LogisticRegression(penalty='l2', dual=True, C=1)
    model.fit(traindata, targets)
    

if __name__ == '__main__':
    
    traindata, testdata, targets_pp, targets = load_data()
    
    # this model uses all features to predict gender per page
    models = train_comb_model(traindata, targets_pp)
    output_comb = models[0].predict_proba(testdata)
    
    # this model combines gender predictions per page into prediction per writer
    fmodel = train_final_model(output_comb, targets)
    output_comb = fmodel.predict_proba(testdata)
    
    np.savetxt(DATA_DIR+'submission.csv', output_comb[:,0], delimiter=",")
    
