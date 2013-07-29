#!/usr/bin/env python

import os
import numpy as np
import pandas as pd
import itertools
from sklearn import preprocessing, cross_validation, ensemble, linear_model
from scipy import sparse
from collections import defaultdict
from Employee import kmodes
from random import Random
import inspyred
from multiprocessing import Pool


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
    keymap = []
    ysums = [defaultdict(int) for _ in range(data.shape[1])]
    for iN, col in enumerate(data.T):
        counts = defaultdict(int)
        for j, el in enumerate(col):
            counts[el] += 1
            if j >= len(y):
                ysums[iN][el] += 0.5
            else:
                ysums[iN][el] += y[el]
        
        uniques = set([x for x in col if counts[x] > ftresh])
        keymap.append(dict((key, i) for i, key in enumerate(uniques)))
    
    outmat = np.empty(data.shape)
    for iN, col in enumerate(data.T):
        km = keymap[iN]
        num_labels = len(km)
        for j, val in enumerate(col):
            if val in km:
                outmat[j, iN] = float(ysums[iN][val]) / num_labels
    return outmat

if __name__ == "__main__":
    
    rseed       = 42
    featDegree  = 3
    cvN         = 5
    ftresh      = 1
    optAlgo     = 'PSO'
    
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
    for cc in clusters:
        allData = np.hstack((allData, np.expand_dims(cc,1)))
    
    print("Combining features...")
    # create higher-order features
    allData = pd.DataFrame(allData)
    if featDegree > 1:
        combData = pd.DataFrame()
        for fd in range(2,featDegree+1):
            combData = pd.concat((combData, construct_combined_features_pd(allData, degree=fd)), ignore_index=True, axis=1)
        allData = pd.concat((allData, combData), ignore_index=True, axis=1)
    allData = np.array(allData)
    
    print("Performing feature selection...")
    numFeatures = allData.shape[1]
    cvModel = linear_model.LogisticRegression(penalty='l2', C=2.0, class_weight='auto', random_state=rseed)
    
    # Xts holds one hots encodings for each individual feature in memory, speeding up feature selection 
    Xts = [selective_onehotencoder(allData[:numTrain,[i]], ftresh)[0] for i in range(numFeatures)]
    
    print("Performing greedy feature selection...")
    scoreHist = []
    greedFeats = set([])
    while len(scoreHist) < 2 or scoreHist[-1][0] > scoreHist[-2][0]:
        scores = []
        for f in range(len(Xts)):
            if f not in greedFeats:
                feats = list(greedFeats) + [f]
                Xt = sparse.hstack([Xts[j] for j in feats]).tocsr()
                score = cv_loop(Xt, y, cvModel, cvN)
                scores.append((score, f))
                print("Feature: %i Mean AUC: %f" % (f, score))
        greedFeats.add(max(scores)[1])
        scoreHist.append(max(scores))
        print("Current features: %s" % sorted(list(greedFeats)))
    greedFeats.remove(scoreHist[-1][1])
    
    print("Performing smart feature selection...")
    prng = Random()
    prng.seed(rseed)
    if optAlgo == 'GA':
        ea = inspyred.ec.EvolutionaryComputation(prng)
        ea.selector = inspyred.ec.selectors.tournament_selection
        ea.variator = [inspyred.ec.variators.partially_matched_crossover, 
                       inspyred.ec.variators.inversion_mutation]
        ea.replacer = inspyred.ec.replacers.generational_replacement
        ea.terminator = inspyred.ec.terminators.generation_termination
        final_pop = ea.evolve(generator=generator_feats,
                              evaluator=evaluator_feats,
                              bounder = inspyred.ec.DiscreteBounder([0, 1]),
                              maximize=True,
                              pop_size=40,
                              max_generations=250,
                              tournament_size=8,
                              num_selected=100,
                              num_elites=1,
                              numFeatures=numFeatures, Xts=Xts, y=y, cvModel=cvModel,cvN=cvN
                              )
        smartFeats = set(max(ea.population))
    elif optAlgo == 'PSO':
        ea = inspyred.swarm.PSO(prng)
        ea.terminator = inspyred.ec.terminators.evaluation_termination
        ea.topology = inspyred.swarm.topologies.ring_topology
        final_pop = ea.evolve(generator=generator_feats,
                              evaluator=evaluator_feats,
                              bounder = inspyred.ec.DiscreteBounder([0, 1]),
                              maximize=True,
                              pop_size=10,
                              max_generations=50,
                              max_evaluations=30000,
                              neighborhood_size=5,
                              numFeatures=numFeatures, Xts=Xts, y=y, cvModel=cvModel,cvN=cvN
                              )
        smartFeats = set(max(ea.population))
    else:
        smartFeats = set([])
    
    allData = allData[:, list(greedFeats)]
    
    print("Converting to one-hot...")
    allData, _ = selective_onehotencoder(allData, ftresh)
    xTrain = allData[:numTrain]
    xTest = allData[numTrain:]
    
    print("Training models...")
    models = [  #ensemble.GradientBoostingClassifier(n_estimators=50, max_depth=2,
                #min_samples_leaf=5, subsample=0.5, max_features=0.25),
                #ensemble.RandomForestClassifier(n_estimators=50, max_depth=4,
                #min_samples_leaf=5, max_features=0.25),
                linear_model.LogisticRegression(penalty='l1', C=2.0, fit_intercept=True,
                intercept_scaling=1, class_weight='auto', random_state=rseed),
                linear_model.LogisticRegression(penalty='l2', C=2.0, fit_intercept=True,
                intercept_scaling=1, class_weight='auto', random_state=rseed),
            ]
    
    cv = cross_validation.StratifiedShuffleSplit(y, random_state=rseed, n_iter=cvN)
    scores = [0]*len(models)
    for ii in range(len(models)):
        if isinstance(models[ii], ensemble.GradientBoostingClassifier) or \
            isinstance(models[ii], ensemble.RandomForestClassifier):
            xTrain = xTrain.toarray()
        scores[ii] = cross_validation.cross_val_score(models[ii], xTrain, y, scoring='roc_auc', n_jobs=1, cv=cv)
        print("Cross-validation AUC on the training set for model %d:" % ii)
        print("%0.3f (+/-%0.03f)" % (scores[ii].mean(), scores[ii].std() / 2))
        
        models[ii].fit(xTrain, y)
    
    print("Making prediction...")
    preds = []
    for ii in range(len(models)):
        if isinstance(models[ii], ensemble.GradientBoostingClassifier) or \
            isinstance(models[ii], ensemble.RandomForestClassifier):
            xTest = xTest.toarray()
        preds.append(models[ii].predict_proba(xTest)[:,1])
    
    create_submission(np.mean(np.array(preds), axis=0), DATA_DIR+'submission.csv')
    
    print("Done.")
