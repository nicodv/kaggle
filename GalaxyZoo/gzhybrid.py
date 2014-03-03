#!/usr/bin/python2

import os
import math
import cv2
import scipy.stats
import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import cross_validation, ensemble, metrics, decomposition, preprocessing
from sklearn.pipeline import Pipeline

DATA_DIR = '/home/nico/Data/GalaxyZoo/'


def extra_feats():
    filenumbers = np.load(os.path.join(DATA_DIR, 'filenumbers.npy'))

    for id, dr in enumerate(['images_training_proc', 'images_test_proc']):
        xfeats = []
        for ii, fn in tqdm.tqdm(enumerate(filenumbers[id])):
            for itype in ('raw', 'proc'):
                curfeats = []
                im = cv2.imread(os.path.join(DATA_DIR, dr, str(fn) + '_' + itype + '.png'))
                # do this on grayscale
                im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                # blur a bit to decrease noise
                im = cv2.GaussianBlur(im, (5, 5), 0)
                w, h = im.shape

                # 4 moments at various horizontal and vertical cross-sections
                for func in (np.mean, np.std, scipy.stats.skew, scipy.stats.kurtosis):
                    curfeats.append(np.mean(np.ravel(im)))
                    curfeats.append(func(im[:, int(w * 1/3)]))
                    curfeats.append(func(im[:, int(w * 2/5)]))
                    curfeats.append(func(im[:, int(w * 1/2)]))
                    curfeats.append(func(im[:, int(w * 3/5)]))
                    curfeats.append(func(im[:, int(w * 2/3)]))
                    curfeats.append(func(im[int(w * 1/3), :]))
                    curfeats.append(func(im[int(w * 2/5), :]))
                    curfeats.append(func(im[int(w * 1/2), :]))
                    curfeats.append(func(im[int(w * 3/5), :]))
                    curfeats.append(func(im[int(w * 2/3), :]))

                # intensity ratio = ratio between 4 moments of intensity in small center square divided by
                # 4 moments of intensity in bigger center square
                for func in (np.mean, np.std, scipy.stats.skew, scipy.stats.kurtosis):
                    ratio = func(im[int(w * 2/5):int(w * 3/5), int(w * 2/5):int(w * 3/5)]) / \
                            func(im[int(w * 1/4):int(w * 3/4), int(w * 1/4):int(w * 3/4)])
                    ratio = np.clip(ratio, 0., 10.)
                    if np.isnan(ratio):
                        ratio = 10.
                    curfeats.append(ratio)

                if not (itype == 'raw' and id == 0):
                    xfeats.append(curfeats)

        if id == 0:
            np.save(os.path.join(DATA_DIR, 'xfeats_train.npy'), np.array(xfeats))
        elif id == 1:
            np.save(os.path.join(DATA_DIR, 'xfeats_test.npy'), np.array(xfeats))
    print("Extra features generated")
    return


def load_data():

    feattrain1 = np.load(DATA_DIR+'feats_convnet_train1.npy')
    feattest1 = np.load(DATA_DIR+'feats_convnet_test1.npy')
    feattrain2 = np.load(DATA_DIR+'feats_convnet_train2.npy')
    feattest2 = np.load(DATA_DIR+'feats_convnet_test2.npy')
    xfeattrain = np.load(DATA_DIR+'xfeats_train.npy')
    xfeattest = np.load(DATA_DIR+'xfeats_test.npy')

    traindata = np.concatenate((feattrain1, feattrain2, xfeattrain), axis=1)
    testdata = np.concatenate((feattest1, feattest2, xfeattest), axis=1)

    targets = np.load(DATA_DIR+'targets.npy')
    targets = np.repeat(targets, 2, axis=0)
    print("Data loaded")

    return traindata, testdata, targets


def train_comb_model(traindata, targets):

    models = [
        # ensemble.GradientBoostingRegressor(n_estimators=50, learning_rate=0.1,
        # max_depth=3, subsample=0.5, max_features='sqrt', min_samples_leaf=250),
        # ensemble.ExtraTreesRegressor(n_estimators=50, max_features='sqrt', max_depth=6,
        # oob_score=False, min_samples_leaf=250, n_jobs=1),
        ensemble.RandomForestRegressor(n_estimators=50, max_features='sqrt', max_depth=8,
        oob_score=False, min_samples_leaf=250, n_jobs=1)
    ]

    cv = cross_validation.ShuffleSplit(len(targets), n_iter=5, train_size=0.8)

    print("Running model(s)")
    scores = [0]*len(models)
    for i in range(len(models)):
        # get scores
        scores[i] = cross_validation.cross_val_score(models[i], traindata, targets,
                                                     cv=cv, n_jobs=-1, scoring='mean_squared_error')
        # calculate RMSE; MSE is negative, so minus
        scores[i] = np.sqrt(-scores[i])
        print("Cross-validation accuracy on the training set for model %d:" % i)
        print("%0.4f (+/-%0.04f)" % (scores[i].mean(), scores[i].std() / 2))

        models[i].fit(traindata, targets)

    return models


def get_preprocessor(data):
    pipe = Pipeline([('scaler', preprocessing.StandardScaler()),
                     ('pca', decomposition.PCA(n_components=100, whiten=True))])
    pipe.fit(data)
    print("Preprocessing finished")
    return pipe


if __name__ == '__main__':

    # extra_feats()

    traindata, testdata, targets = load_data()
    fn = np.load(os.path.join(DATA_DIR, 'filenumbers.npy'))[1]

    # preprocess data
    preproc = get_preprocessor(traindata)
    proc_train = preproc.transform(traindata)
    proc_test = preproc.transform(testdata)

    # now define and train model
    models = train_comb_model(proc_train, targets)

    output = models[0].predict(proc_test)

    # save test output as submission
    subm = pd.DataFrame(np.hstack((np.array(fn).reshape([-1,1]), output)))
    subm.columns = ['GalaxyId', 'Class1.1', 'Class1.2', 'Class1.3', 'Class2.1', 'Class2.2',
                    'Class3.1', 'Class3.2', 'Class4.1', 'Class4.2',
                    'Class5.1', 'Class5.2', 'Class5.3', 'Class5.4', 'Class6.1', 'Class6.2',
                    'Class7.1', 'Class7.2', 'Class7.3', 'Class8.1', 'Class8.2', 'Class8.3', 'Class8.4', 'Class8.5', 'Class8.6', 'Class8.7',
                    'Class9.1', 'Class9.2', 'Class9.3', 'Class10.1', 'Class10.2', 'Class10.3',
                    'Class11.1', 'Class11.2', 'Class11.3', 'Class11.4', 'Class11.5', 'Class11.6']

    subm.to_csv(DATA_DIR+'model_hybrid.csv', header=True, index=False)
