#!/usr/bin/python

import numpy as np
import scipy
from sklearn import cross_validation, ensemble, metrics, linear_model

DATA_DIR = '/home/nico/datasets/Kaggle/GenderWrite/'

def load_data():
    mtrain = np.load(DATA_DIR+'maxout_train.npy')
    mtest = np.load(DATA_DIR+'maxout_test.npy')
    feattrain = np.load(DATA_DIR+'trainfeat.npy')
    feattest = np.load(DATA_DIR+'testfeat.npy')
    
    traindata = np.concatenate((mtrain, feattrain), axis=1)
    testdata = np.concatenate((mtest, feattest), axis=1)
    
    targets = np.load(DATA_DIR+'targets.npy')[:,0]
    
    return traindata, testdata, targets

def train_model(traindata, targets):
    
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
    
