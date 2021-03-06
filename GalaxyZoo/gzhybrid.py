#!/usr/bin/python2

import os
import math
import cv2
import scipy.stats
import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import cross_validation, ensemble, metrics, decomposition, preprocessing, linear_model, naive_bayes
from sklearn.pipeline import Pipeline

DATA_DIR = '/home/nico/Data/GalaxyZoo/'


def extra_feats():
    filenumbers = np.load(os.path.join(DATA_DIR, 'filenumbers.npy'))

    for id, dr in enumerate(['images_training_proc', 'images_test_proc']):
        xfeats = []
        for ii, fn in tqdm.tqdm(enumerate(filenumbers[id])):
            for itype in ('raw', 'proc'):
                # only for training do we actually use the raw data
                if not (itype == 'raw' and id == 1):
                    curfeats = []
                    im = cv2.imread(os.path.join(DATA_DIR, dr, str(fn) + '_' + itype + '.png'))

                    # RGB
                    meanint = np.mean(im)
                    for cid in range(3):
                        curfeats.append(np.mean(im[:, :, cid]) - meanint)

                    # do rest on grayscale
                    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                    # blur a bit to decrease noise
                    im = cv2.GaussianBlur(im, (3, 3), 0)
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
                        ratio = func(np.ravel(im[int(w * 2/5):int(w * 3/5), int(w * 2/5):int(w * 3/5)])) / \
                                func(np.ravel(im[int(w * 1/4):int(w * 3/4), int(w * 1/4):int(w * 3/4)]))
                        ratio = np.clip(ratio, 0., 10.)
                        if np.isnan(ratio):
                            ratio = 10.
                        curfeats.append(ratio)

                    # circle of pixels at various radii
                    circles = []
                    for r in (5, 10, 15, 20, 30, 40, 50):
                        theta = np.linspace(0.5 * np.pi, 1.5 * np.pi, 5 * r)
                        xy = list(set([xy for xy in zip((r * np.cos(theta) - int(w/2)),
                                                        (r * np.sin(theta) - int(h/2)))]))
                        circles.append([im[int(x), int(y)] for (x, y) in xy])
                        for func in (np.mean, np.std, scipy.stats.skew, scipy.stats.kurtosis):
                            curfeats.append(func(circles[-1]))

                    # now count how many peaks and throughs we see in that matrix
                    # this is useful for determining no. of spiral arms and tightness of spiral
                    for c in circles:
                        # smooth out a bit
                        for ii, el in enumerate(c):
                            if ii == 0:
                                c[ii] = np.mean(c[:2] + [c[-1]])
                            elif ii == len(c) - 1:
                                c[ii] = np.mean(c[ii-1:] + [c[0]])
                            else:
                                c[ii] = np.mean(c[ii-1:ii+2])

                        nc = (c - np.mean(c)) / np.std(c)

                        inpeak = False
                        peakcntr = 0
                        intrough = True
                        troughcntr = 0
                        lastx = 0
                        for el in nc:
                            if el < -1.5:
                                intrough = True
                            else:
                                if intrough and lastx == 2:
                                    troughcntr += 1
                                    lastx = 1
                                intrough = False
                            if el > 1.5:
                                inpeak = True
                            else:
                                if inpeak and lastx == 1:
                                    peakcntr += 1
                                    lastx = 2
                                inpeak = False

                        curfeats.append(min(troughcntr, peakcntr))

                    xfeats.append(curfeats)

        if id == 0:
            np.save(os.path.join(DATA_DIR, 'xfeats_train.npy'), np.array(xfeats))
        elif id == 1:
            np.save(os.path.join(DATA_DIR, 'xfeats_test.npy'), np.array(xfeats))
    print("Extra features generated")
    return


def load_data():

    print("Loading data")

    tr, te = [], []
    for trd, ted in zip(('feats_daex_train1.npy',
                        ),
                        ('feats_daex_test1.npy',
                        )):
        tr.append(np.load(DATA_DIR+trd))
        te.append(np.load(DATA_DIR+ted))
        # delete the raw-features-based items of the test set (not actually going to use them)
        te[-1] = te[-1][1::2, :]

    tr.append(np.load(DATA_DIR+'xfeats_train.npy'))
    te.append(np.load(DATA_DIR+'xfeats_test.npy'))

    traindata = np.concatenate(tr, axis=1)
    testdata = np.concatenate(te, axis=1)

    targets = np.load(DATA_DIR+'targets.npy')
    targets = np.repeat(targets, 2, axis=0)

    return traindata, testdata, targets


def train_comb_model(traindata, targets):

    model = ensemble.RandomForestRegressor(n_estimators=50, max_depth=10, max_features='sqrt', min_samples_leaf=100,
                                           n_jobs=-1)

    cv = cross_validation.ShuffleSplit(len(targets), n_iter=5, train_size=0.6)

    print("Cross-validating model")
    # get scores
    scores = cross_validation.cross_val_score(model, traindata, targets,
                                              cv=cv, n_jobs=1, scoring='mean_squared_error')
    # calculate RMSE; MSE is negative, so minus
    scores = np.sqrt(-scores)
    print("RMSE on the training set:")
    print("%0.4f (+/-%0.04f)" % (scores.mean(), scores.std() / 2))

    print("Training model")
    model.fit(traindata, targets)

    return model


def get_preprocessor(data):
    print("Preprocessing")
    pipe = Pipeline([('mmscaler', preprocessing.MinMaxScaler(feature_range=(0, 1))),
                     # ('scaler', preprocessing.StandardScaler()),
                     # ('pca', decomposition.PCA(n_components=100, whiten=True)),
                     ('nnmf', decomposition.NMF(n_components=100))])
    pipe.fit(data)
    return pipe


if __name__ == '__main__':

    # extra_feats()

    traindata, testdata, targets = load_data()

    # preprocess data
    # preproc = get_preprocessor(traindata)
    # proc_train = preproc.transform(traindata)
    # testdata = preproc.transform(testdata)

    # now define and train model
    model = train_comb_model(traindata, targets)

    print("Making a prediction")
    output = model.predict(testdata)

    # save test output as submission
    fn = np.load(os.path.join(DATA_DIR, 'filenumbers.npy'))[1]
    fn = np.reshape(np.array([int(el) for el in fn]), (-1, 1))
    output = np.hstack((fn, output))

    np.savetxt(DATA_DIR+'model_hybrid_daextest.csv', output, delimiter=',', comments='',
               fmt=["%6d"] + ["%8.6f"] * 37,
               header='GalaxyId,Class1.1,Class1.2,Class1.3,Class2.1,Class2.2,Class3.1,Class3.2,Class4.1,Class4.2,' +
                      'Class5.1,Class5.2,Class5.3,Class5.4,Class6.1,Class6.2,Class7.1,Class7.2,Class7.3,Class8.1,' +
                      'Class8.2,Class8.3,Class8.4,Class8.5,Class8.6,Class8.7,Class9.1,Class9.2,Class9.3,Class10.1,' +
                      'Class10.2,Class10.3,Class11.1,Class11.2,Class11.3,Class11.4,Class11.5,Class11.6')
