#!/usr/bin/python

import numpy as np
from sklearn import cross_validation, ensemble, metrics

DATA_DIR = '/home/nico/datasets/Kaggle/Whales/'

def load_data():
    outtrain = np.load(DATA_DIR+'convout_train.npy')
    outtest = np.load(DATA_DIR+'convout_test.npy')
    
    spectrain = np.load(DATA_DIR+'trainspecfeat.npy')
    spectrain = np.concatenate((np.mean(spectrain, axis=1),np.std(spectrain, axis=1)), axis=1)
    spectest = np.load(DATA_DIR+'testspecfeat.npy')
    spectest = np.concatenate((np.mean(spectest, axis=1),np.std(spectest, axis=1)), axis=1)
    
#    return np.concatenate((outtrain, spectrain), axis=1), \
#            np.concatenate((outtest, spectest), axis=1)
    return spectrain, spectest

def train_model(traindata, targets):
    
    models = [
                ensemble.GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, \
                max_depth=4, subsample=0.5, max_features=3, min_samples_leaf=20)
                ]
    
    # use StratifiedKFold, because survived 0/1 is not evenly distributed
    cv = cross_validation.StratifiedKFold(targets, n_folds=10)
    
    scores = [0]*len(models)
    for i in range(len(models)):
        # get scores
        scores[i] = cross_validation.cross_val_score(models[i], traindata, targets, \
                    cv=cv, n_jobs=1, score_func=metrics.accuracy_score)
        print "Cross-validation accuracy on the training set for model %d:" % i
        print "%0.3f (+/-%0.03f)" % (scores[i].mean(), scores[i].std() / 2)
        
        models[i].fit(traindata, targets)
    
    return models

if __name__ == '__main__':
    
    traindata, testdata = load_data()
    
    targets = np.load(DATA_DIR+'targets.npy')[:,0]
    
    models = train_model(traindata, targets)
    
    output = models[0].predict_proba(testdata)
    np.savetxt(DATA_DIR+'model_gbm.csv', output[:,0], delimiter=",")
    