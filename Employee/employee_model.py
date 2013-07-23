#!/usr/bin/env python

import os
import numpy as np
import pandas as pd
import itertools
from sklearn import preprocessing, cross_validation, ensemble, metrics
from collections import defaultdict
from scipy import sparse
import kmodes

DATA_DIR = '/home/nico/datasets/Kaggle/Employee/'
TRAIN_FILE = DATA_DIR+'train.csv'
TEST_FILE = DATA_DIR+'test.csv'
CLUST_FILE = DATA_DIR+'clusters.npy'

SEED = 42

def construct_combined_features(data, N=3):
    '''Combine features into a set of new features that express the
    2nd to Nth degree combinations of original features.
    '''
    new_data = []
    sources = []
    _, nfeat = data.shape
    for degree in range(2,N+1):
        for indices in itertools.combinations(range(nfeat), degree):
            new_data.append([hash(tuple(v)) for v in data[:,indices]])
            sources.append(indices)
    return np.array(new_data).T, np.array(sources)

def create_submission(predictions, filename):
    print("Saving submission...")
    with open(filename, 'w') as f:
        f.write("id,ACTION\n")
        for i, pred in enumerate(predictions):
            f.write("%d,%f\n" % (i + 1, pred))

def cv_loop(X, y, model, rseed=SEED, n_iter=5):
    cv = cross_validation.StratifiedShuffleSplit(y, random_state=rseed, n_iter=n_iter)
    scores = cross_validation.cross_val_score(model, X, y, scoring='roc_auc', n_jobs=4, cv=cv)
    return np.mean(scores)

if __name__ == "__main__":
    
    N = 5
    
    print("Reading data...")
    trainData = pd.read_csv(TRAIN_FILE)
    testData = pd.read_csv(TEST_FILE)
    y = np.array(trainData.ACTION)
    
    # last column is duplicate, so dropped here
    numTrain = np.shape(trainData)[0]
    allData = np.vstack((trainData.ix[:,1:-1], testData.ix[:,1:-1]))
    
    print("Transforming data...")
    combData, sources = construct_combined_features(allData, N=3)
    allData = np.hstack((allData, combData))
    numFeatures = allData.shape[1]
    
    print("Dropping rare features...")
    counts = defaultdict(int)
    for feat in allData.T:
        counts[tuple(feat)] += 1
    treshold = 5
    allData = allData[:,counts > treshold]
    
    if os.path.exist(DATA_DIR+'clusters'):
        print("Loading clusters...")
        clusters = np.load(CLUST_FILE)
    else:
        print("Starting cluster analysis...")
        Xclust, cent = kmodes.kmodes(allData, k=4, maxiters=100)
    
    print("Performing feature selection...")
    cvModel = ensemble.GradientBoostingClassifier(n_estimators=40, max_depth=5,
                min_samples_leaf=10, subsample=0.5, max_features=0.25)
    # replace predict so that we can use AUC in cross-validation
    cvModel.predict = lambda m, x: m.predict_proba(x)[:,1]
    
    print("Performing smart feature selection...")
    
    print("Performing greedy feature selection...")
    # Xts holds one hot encodings for each individual feature in memory
    # speeding up feature selection 
    Xts = [preprocessing.OneHotEncoder(allData[:numTrain,[i]])[0] for i in range(numFeatures)]
    scoreHist = []
    goodFeats = set([])
    while len(scoreHist) < 2 or scoreHist[-1][0] > scoreHist[-2][0]:
        scores = []
        for f in range(len(Xts)):
            if f not in goodFeats:
                feats = list(goodFeats) + [f]
                Xt = sparse.hstack([Xts[j] for j in feats]).tocsr()
                score = cv_loop(Xt, y, cvModel, N)
                scores.append((score, f))
                print("Feature: %i Mean AUC: %f" % (f, score))
        goodFeats.add(max(scores)[1])
        scoreHist.append(max(scores))
        print("Current features: %s" % sorted(list(goodFeats)))
    goodFeats.remove(scoreHist[-1][1])
    goodFeats = sorted(list(goodFeats))
    
    allData = allData[:, goodFeats]
    
    print("Converting to one-hot...")
    ohEncoder = preprocessing.OneHotEncoder()
    ohEncoder.fit(allData)
    xTrain = ohEncoder.transform(allData[:numTrain])
    xTest = ohEncoder.transform(allData[numTrain:])
    
    print("Training models...")
    models = [ensemble.GradientBoostingClassifier(n_estimators=40, max_depth=5,
                min_samples_leaf=10, subsample=0.5, max_features=0.25),
            ]
    
    cv = cross_validation.StratifiedShuffleSplit(y, random_state=SEED, n_iter=N)
    scores = [0]*len(models)
    for ii in range(len(models)):
        scores[i] = cross_validation.cross_val_score(models[ii], xTrain, y, scoring='roc_auc', n_jobs=4, cv=cv)
        print("Cross-validation AUC on the training set for model %d:" % ii)
        print("%0.3f (+/-%0.03f)" % (scores[ii].mean(), scores[ii].std() / 2))
        
        models[ii].fit(xTrain, y)
    
    print("Making prediction...")
    preds = model.predict(xTest)
    create_submission(preds, DATA_DIR+'submission.csv')
    
    print("Done.")
