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

DATA_DIR = '/home/nico/datasets/Kaggle/Employee/'
TRAIN_FILE = DATA_DIR+'train.csv'
TEST_FILE = DATA_DIR+'test.csv'
FEAT_FILE = DATA_DIR+'feats.npy'
CLUST_FILE = DATA_DIR+'clusters.npy'

def construct_combined_features(data, degree=2):
    '''Combine features into a set of new features that express the
    nth degree combinations of original features.
    '''
    new_data = []
    _, nfeat = data.shape
    for indices in itertools.combinations(range(nfeat), degree):
        new_data.append([hash(tuple(v)) for v in data[:,indices]])
    return np.array(new_data).T

def create_submission(predictions, filename):
    print("Saving submission...")
    with open(filename, 'w') as f:
        f.write("id,ACTION\n")
        for i, pred in enumerate(predictions):
            f.write("%d,%f\n" % (i + 1, pred))

def cv_loop(X, y, model, rseed=42, n_iter=8):
    cv = cross_validation.StratifiedShuffleSplit(y, random_state=rseed, n_iter=n_iter)
    scores = cross_validation.cross_val_score(model, X, y, scoring='roc_auc', n_jobs=4, cv=cv)
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
        mask = []
        for comp in cs:
            mask.append(comp.element[0])
        Xt = sparse.hstack([Xts[j] for j in mask]).tocsr()
        fit = cv_loop(Xt, y, cvModel, cvN)
        fitness.append(fit)
    return fitness

if __name__ == "__main__":
    
    rseed       = 42
    featDegree  = 2
    cvN         = 4
    ftresh      = 10
    optAlgo     = 'none'
    
    print("Reading data...")
    trainData = pd.read_csv(TRAIN_FILE)
    testData = pd.read_csv(TEST_FILE)
    y = np.array(trainData.ACTION)
    
    # last column is duplicate, so dropped here
    numTrain = np.shape(trainData)[0]
    allData = np.vstack((trainData.ix[:,1:-1], testData.ix[:,1:-1]))
    
    if os.path.exists(CLUST_FILE):
        print("Loading clusters...")
        clusters = np.load(CLUST_FILE)
    else:
        print("Starting cluster analysis...")
        clusters = []
        for k in (10, 30, 100, 300, 1000):
            #cc, _, _ = kmodes.opt_kmodes(allData, k, init='Cao', preRuns=10, goodPctl=20,
                                         #centUpd='wsample', maxIters=200)
            cc, _, _ = kmodes.kmodes(allData, k, init='Cao', centUpd='mode', maxIters=100)
            clusters.append(cc)
        np.save(CLUST_FILE, clusters)
    for cc in clusters:
        allData = np.hstack((allData, np.expand_dims(cc,1)))
    
    print("Combining features...")
    # create higher-order features
    if featDegree > 1:
        for fd in range(2,featDegree+1):
            combData = construct_combined_features(allData, degree=fd)
            allData = np.hstack((allData, combData))
    
    numFeatures = allData.shape[1]
    
    print("Consolidating rare features...")
    rare = np.empty((allData.shape[1], allData.shape[0]), dtype='bool')
    for ii, feat in enumerate(allData.T):
        counts = defaultdict(int)
        for el in feat:
            counts[el] += 1
        rare[ii] = [counts[x] <= ftresh for x in feat]
    rare = rare.T
    # put all rare values to unique groups with 0 as values
    allData = np.where(rare, allData, -1)
    allData += 1
    
    print("Performing feature selection...")
    cvModel = ensemble.GradientBoostingClassifier(n_estimators=40, max_depth=5,
                min_samples_leaf=10, subsample=0.5, max_features=0.25)
    
    # Xts holds one hots encodings for each individual feature in memory, speeding up feature selection 
    Xts = [preprocessing.OneHotEncoder().fit_transform(allData[:numTrain,[i]]) for i in range(numFeatures)]
    
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
                              pop_size=10,
                              max_generations=50,
                              tournament_size=5,
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
    bestFeats = greedFeats.update(smartFeats)
    bestFeats = sorted(list(bestFeats))
    
    allData = allData[:, bestFeats]
    
    print("Converting to one-hot...")
    ohEncoder = preprocessing.OneHotEncoder()
    ohEncoder.fit(allData)
    xTrain = ohEncoder.transform(allData[:numTrain])
    xTest = ohEncoder.transform(allData[numTrain:])
    
    print("Training models...")
    models = [  ensemble.GradientBoostingClassifier(n_estimators=50, max_depth=10,
                min_samples_leaf=5, subsample=0.5, max_features=0.25),
                ensemble.RandomForestClassifier(n_estimators=50, max_depth=10,
                min_samples_leaf=5, max_features=0.25),
                linear_model.LogisticRegression(penalty='l1', C=1.0, fit_intercept=True,
                intercept_scaling=1, class_weight='auto', random_state=rseed),
                linear_model.LogisticRegression(penalty='l2', C=1.0, fit_intercept=True,
                intercept_scaling=1, class_weight='auto', random_state=rseed),
            ]
    
    cv = cross_validation.StratifiedShuffleSplit(y, random_state=rseed, n_iter=cvN)
    scores = [0]*len(models)
    for ii in range(len(models)):
        scores[i] = cross_validation.cross_val_score(models[ii], xTrain, y, scoring='roc_auc', n_jobs=4, cv=cv)
        print("Cross-validation AUC on the training set for model %d:" % ii)
        print("%0.3f (+/-%0.03f)" % (scores[ii].mean(), scores[ii].std() / 2))
        
        models[ii].fit(xTrain, y)
    
    print("Making prediction...")
    preds = model.predict_proba(xTest)
    create_submission(preds, DATA_DIR+'submission.csv')
    
    print("Done.")
