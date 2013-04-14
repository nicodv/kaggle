#!/usr/bin/python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import cross_validation, ensemble, metrics, linear_model

DATA_DIR = '/home/nico/datasets/Kaggle/GenderWrite/'

def load_data():
#    mtrain = np.load(DATA_DIR+'maxout_train.npy')
#    mtest = np.load(DATA_DIR+'maxout_test.npy')
    
    feattrain = np.load(DATA_DIR+'feat_train.npy')
    feattest = np.load(DATA_DIR+'feat_test.npy')
    
#    traindata = np.concatenate((mtrain, feattrain), axis=1)
#    testdata = np.concatenate((mtest, feattest), axis=1)
    
    targets_qpp = np.load(DATA_DIR+'targets_per_page_extra.npy')[:,0]
    targets_pp = np.load(DATA_DIR+'targets_per_page.npy')[:,0]
    targets = np.load(DATA_DIR+'targets.npy')[:,0]
    
    return feattrain, feattest, targets_qpp, targets_pp, targets
#    return traindata, testdata, targets_pp, targets

def train_comb_model(traindata, targets):
    
    models = [
                ensemble.GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, \
                max_depth=3, subsample=0.5, max_features=10, min_samples_leaf=40)
#                ensemble.RandomForestClassifier(n_estimators=500, max_features='log2', max_depth=10, \
#                compute_importances=True, oob_score=False, min_samples_leaf=20, criterion='entropy')
                ]
    
    cv = cross_validation.StratifiedShuffleSplit(targets, n_iter=4)
    
    scores = [0]*len(models)
    for i in range(len(models)):
        # get scores
        scores[i] = cross_validation.cross_val_score(models[i], traindata, targets, \
                    cv=cv, n_jobs=-1, score_func=metrics.auc_score)
        print "Cross-validation accuracy on the training set for comb. model %d:" % i
        print "%0.3f (+/-%0.03f)" % (scores[i].mean(), scores[i].std() / 2)
        
        models[i].fit(traindata, targets)
    
    return models

def train_final_model(traindata, targets):
    model = linear_model.LogisticRegression(penalty='l2', dual=True, C=0.1, fit_intercept=True)
    cv = cross_validation.StratifiedShuffleSplit(targets, n_iter=4)
    scores = cross_validation.cross_val_score(model, traindata, targets, \
        cv=cv, n_jobs=-1, score_func=metrics.auc_score)
    print "Cross-validation accuracy on the training set for final model:"
    print "%0.3f (+/-%0.03f)" % (scores.mean(), scores.std() / 2)
    model.fit(traindata, targets)
    return model

def logloss(yhat, y):
    return -(1./y.shape[0]) * (np.sum(y*np.log(yhat) + (1-y)*np.log(1-yhat)))


if __name__ == '__main__':
    
    traindata, testdata, targets_qpp, targets_pp, targets = load_data()
    
    # this model uses all features to predict gender per page
    comb_model = train_comb_model(traindata, targets_qpp)
    output_comb_train = comb_model[0].predict_proba(traindata)[:,1]
    print "log loss over training data A: %f" % logloss(output_comb_train,targets_qpp)
    output_comb_test = comb_model[0].predict_proba(testdata)[:,1]
    
    # this model combines gender predictions per page into prediction per writer
    output_comb_train = np.reshape(output_comb_train,[output_comb_train.shape[0]/(4*9),-1])
    output_comb_test = np.reshape(output_comb_test,[output_comb_test.shape[0]/(4*9),-1])
    final_model = train_final_model(output_comb_train, targets)
    output_final_train = final_model.predict_proba(output_comb_train)[:,1]
    print "log loss over training data B: %f" % logloss(output_final_train,targets)
    output_final_test = final_model.predict_proba(output_comb_test)[:,1]
    
    f, axarr = plt.subplots(2, 2)
    axarr[0, 0].plot(output_comb_train[output_comb_train.mean(axis=1).argsort()])
    axarr[0, 1].plot(output_comb_test[output_comb_test.mean(axis=1).argsort()])
    axarr[1, 0].plot(output_final_train[output_final_train.argsort()])
    axarr[1, 1].plot(output_final_test[output_final_test.argsort()])
    
    # add writers column
    writers = pd.Series(range(283,476))
    submission_df = pd.DataFrame({'writers': writers, 'y': output_final_test})
    submission_df.to_csv(DATA_DIR+'submission.csv', index=False, header=False)
    
