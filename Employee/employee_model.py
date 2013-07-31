#!/usr/bin/env python

import os
import numpy as np
import pandas as pd
import itertools
from sklearn import preprocessing, cross_validation, ensemble, linear_model
from scipy import sparse
from collections import defaultdict
from Employee import kmodes


DATA_DIR = '/home/nico/datasets/Kaggle/Employee/'
TRAIN_FILE = DATA_DIR+'train.csv'
TEST_FILE = DATA_DIR+'test.csv'
FEAT_FILE = DATA_DIR+'feats.npy'
CLUST_FILE = DATA_DIR+'clusters.npy'

def construct_combined_features_pd(data, degree=2):
    '''Combine features into a set of new features that express the
    nth degree combinations of original features.
    '''
    new_data = []
    _, nfeat = data.shape
    for indices in itertools.combinations(range(nfeat), degree):
        group_ids = data.groupby(list(data.columns[list(indices)])).grouper.group_info[0]
        new_data.append(group_ids)
    return pd.DataFrame(new_data).transpose()

def construct_combined_features(data, degree=2):
    new_data = []
    m,n = data.shape
    for indicies in itertools.combinations(range(n), degree):
        new_data.append([hash(tuple(v)) for v in data[:,indicies]])
    return np.array(new_data).T

def create_submission(preds, filename):
    print("Saving submission...")
    with open(filename, 'w') as f:
        f.write("id,ACTION\n")
        for i, pred in enumerate(preds):
            f.write("%d,%f\n" % (i + 1, pred))

def cv_loop(X, y, model, rseed=42, n_iter=8):
    cv = cross_validation.StratifiedShuffleSplit(y, random_state=rseed, n_iter=n_iter)
    scores = cross_validation.cross_val_score(model, X, y, scoring='roc_auc', n_jobs=1, cv=cv)
    return np.mean(scores)

# inspyred helper methods
def generator_feats(random, args):
    size = args.get('numFeatures')
    return [random.randint(0,1) for i in range(size)]

def evaluator_feats(candidates, args):
    Xts = args.get('Xts')
    y = args.get('y')
    cvModel = args.get('cvModel')
    cvN = args.get('cvN')
    fitness = []
    for cs in candidates:
        Xt = sparse.hstack([Xts[j] for j in cs if j == 1]).tocsr()
        fit = cv_loop(Xt, y, cvModel, cvN)
        fitness.append(fit)
    return fitness

def selective_onehotencoder(data, ftresh):
    keymap = []
    for col in data.T:
        counts = defaultdict(int)
        for el in col:
            counts[el] += 1
        
        uniques = set([x for x in col if counts[x] > ftresh])
        keymap.append(dict((key, i) for i, key in enumerate(uniques)))
    
    total_pts = data.shape[0]
    outdat = []
    for iN, col in enumerate(data.T):
        km = keymap[iN]
        num_labels = len(km)
        spmat = sparse.lil_matrix((total_pts, num_labels))
        for j, val in enumerate(col):
            if val in km:
                spmat[j, km[val]] = 1
        outdat.append(spmat)
    outdat = sparse.hstack(outdat).tocsr()
    return outdat, keymap

def selective_ycoldencoder(data, y, ftresh):
    counts = [defaultdict(int) for _ in range(data.shape[1])]
    ysums = [defaultdict(int) for _ in range(data.shape[1])]
    for iN, col in enumerate(data.T):
        for j, el in enumerate(col):
            if j >= len(y):
                ysums[iN][el] += 0.5
            else:
                ysums[iN][el] += y[j]
            counts[iN][el] += 1
    
    outmat = np.ones(data.shape) * 0.5
    for iN, col in enumerate(data.T):
        for j, val in enumerate(col):
            if counts[iN][val]:
                outmat[j, iN] = float(ysums[iN][val]) / counts[iN][val]
    return outmat

if __name__ == "__main__":
    
    rseed       = 42
    featDegree  = 2
    cvN         = 10
    ftresh      = 0
    optAlgo     = 'GA'
    
    print("Reading data...")
    trainData = pd.read_csv(TRAIN_FILE)
    testData = pd.read_csv(TEST_FILE)
    y = np.array(trainData.ACTION)
    
    # last column is duplicate, so dropped here
    numTrain = np.shape(trainData)[0]
    allData = np.vstack((trainData.ix[:,1:-1], testData.ix[:,1:-1]))
    
    print("Starting cluster analysis...")
    #clusters = []
    #for k in (40, 150, 400, 1500, 4000):
    #    clust = kmodes.opt_kmodes(k, allData, init='Huang', preRuns=10, goodPctl=20, maxIters=200)
    #    #clust = kmodes.KModes(k)
    #    #clust.cluster(allData, init='Huang', maxIters=200)
    #    clusters.append(clust.Xclust)
    #np.save(DATA_DIR+'huang.npy', clusters)
    clusters = np.load(DATA_DIR+'huang.npy')
    for cc in clusters[1:]:
        allData = np.hstack((allData, np.expand_dims(cc,1)))
    
    print("Combining features...")
    # create higher-order features
    allSparseData = pd.DataFrame(allData)
    if featDegree > 1:
        combData = pd.DataFrame()
        for fd in range(2,featDegree+1):
            combData = pd.concat((combData, construct_combined_features_pd(allSparseData, degree=fd)), ignore_index=True, axis=1)
        allSparseData = pd.concat((allSparseData, combData), ignore_index=True, axis=1)
    allSparseData = np.array(allSparseData)
    
    print("Performing feature selection...")
    numSparseFeatures = allSparseData.shape[1]
    cvModel = linear_model.LogisticRegression(penalty='l2', C=4.0, class_weight='auto', random_state=rseed)
    
    # Xts holds one hots encodings for each individual feature in memory, speeding up feature selection 
    Xts = [selective_onehotencoder(allSparseData[:numTrain,[i]], ftresh)[0] for i in range(numSparseFeatures)]
    
    print("Performing greedy feature selection...")
    scoreHist = []
    greedSparseFeats = set([])
    while len(scoreHist) < 2 or scoreHist[-1][0] > scoreHist[-2][0]:
        scores = []
        for f in range(len(Xts)):
            if f not in greedSparseFeats:
                feats = list(greedSparseFeats) + [f]
                Xt = sparse.hstack([Xts[j] for j in feats]).tocsr()
                score = cv_loop(Xt, y, cvModel, cvN)
                scores.append((score, f))
                print("Feature: %i Mean AUC: %f" % (f, score))
        greedSparseFeats.add(max(scores)[1])
        scoreHist.append(max(scores))
        print("Current features: %s" % sorted(list(greedSparseFeats)))
    greedSparseFeats.remove(scoreHist[-1][1])
    #tresh=1, degree=4, huangclusters[1:] --> 0.89
    #greedSparseFeats = [0, 8, 78, 86, 88, 111, 113, 122, 169, 263, 302, 310, 336, 345, 356, 357, 368, 384, 396, 418, 424, 449, 506, 527, 533, 621, 711]
    #tresh=1, degree=2, huangcluster[1:] --> 0.88
    #greedSparseFeats = [0, 8, 12, 14, 15, 16, 22, 27, 30, 32, 57, 67]
    allSparseData = allSparseData[:, list(greedSparseFeats)]
    
    print("Converting to one-hot...")
    allSparseData, _ = selective_onehotencoder(allSparseData, ftresh)
    xSparseTrain = allSparseData[:numTrain]
    xSparseTest = allSparseData[numTrain:]
    allData = selective_ycoldencoder(allData, y, ftresh)
    xTrain = allData[:numTrain]
    xTest = allData[numTrain:]
    
    print("Training models...")
    modelsSparse = [
                linear_model.LogisticRegression(penalty='l2', C=1.0, fit_intercept=False,
                class_weight='auto', random_state=rseed),
                linear_model.LogisticRegression(penalty='l2', C=2.0, fit_intercept=False,
                class_weight='auto', random_state=rseed),
                linear_model.LogisticRegression(penalty='l2', C=4.0, fit_intercept=True,
                class_weight='auto', random_state=rseed),
                linear_model.LogisticRegression(penalty='l2', C=8.0, fit_intercept=True,
                class_weight='auto', random_state=rseed),
                linear_model.LogisticRegression(penalty='l2', C=12.0, fit_intercept=True,
                class_weight='auto', random_state=rseed)
            ]
    models = [
                ensemble.GradientBoostingClassifier(n_estimators=50, max_depth=4,
                min_samples_leaf=5, subsample=0.5, max_features=0.25),
                ensemble.GradientBoostingClassifier(n_estimators=100, max_depth=3,
                min_samples_leaf=10, subsample=0.5, max_features=0.25),
                ensemble.GradientBoostingClassifier(n_estimators=200, max_depth=2,
                min_samples_leaf=20, subsample=0.5, max_features=0.25),
            ]
    
    cv = cross_validation.StratifiedShuffleSplit(y, random_state=rseed, n_iter=cvN)
    scores = [0]*len(modelsSparse)
    for ii in range(len(modelsSparse)):
        scores[ii] = cross_validation.cross_val_score(modelsSparse[ii], xSparseTrain, y, scoring='roc_auc', n_jobs=1, cv=cv)
        print("Cross-validation AUC on the training set for model %d:" % ii)
        print("%0.3f (+/-%0.03f)" % (scores[ii].mean(), scores[ii].std() / 2))
        
        modelsSparse[ii].fit(xSparseTrain, y)
    
    cv = cross_validation.StratifiedShuffleSplit(y, random_state=rseed, n_iter=cvN)
    scores = [0]*len(models)
    for ii in range(len(models)):
        scores[ii] = cross_validation.cross_val_score(models[ii], xTrain, y, scoring='roc_auc', n_jobs=1, cv=cv)
        print("Cross-validation AUC on the training set for model %d:" % ii)
        print("%0.3f (+/-%0.03f)" % (scores[ii].mean(), scores[ii].std() / 2))
        
        models[ii].fit(xTrain, y)
    
    print("Making prediction...")
    preds = []
    for ii in range(len(models)):
        preds.append(models[ii].predict_proba(xTest)[:,1])
    
    create_submission(np.mean(np.array(preds), axis=0), DATA_DIR+'submission.csv')
    
    print("Done.")

#import inspyred
#from random import Random
#print("Performing smart feature selection...")
#prng = Random()
#prng.seed(rseed)
#ea = inspyred.ec.EvolutionaryComputation(prng)
#ea.selector = inspyred.ec.selectors.tournament_selection
#ea.variator = [inspyred.ec.variators.partially_matched_crossover, 
#               inspyred.ec.variators.inversion_mutation]
#ea.replacer = inspyred.ec.replacers.generational_replacement
#ea.terminator = inspyred.ec.terminators.generation_termination
#final_pop = ea.evolve(generator=generator_feats,
#                      evaluator=evaluator_feats,
#                      bounder = inspyred.ec.DiscreteBounder([0, 1]),
#                      maximize=True,
#                      pop_size=32,
#                      max_generations=250,
#                      num_selected=24,
#                      tournament_size=4,
#                      num_elites=1,
#                      numFeatures=numSparseFeatures, Xts=Xts, y=y, cvModel=cvModel,cvN=cvN
#                      )
#smartFeats = np.nonzero(max(ea.population).candidate)[0]
