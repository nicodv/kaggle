#!/usr/bin/env python

import sys
import random
import time

import numpy as np
import pandas as pd
import json

from sklearn import feature_extraction, metrics, linear_model
from sklearn.cross_validation import KFold
from scipy import sparse
from itertools import combinations
import kmodes


BASE_PATH = '/home/nico/datasets/Kaggle/Evergreen/'


def dict_encode(encoding, value):
    if not value in encoding:
        encoding[value] = {'code': len(encoding) + 1, 'count': 0}
    enc = encoding[value]
    enc['count'] += 1
    encoding[value] = enc
    return encoding


def dict_decode(encoding, value, min_occurs):
    enc = encoding[value]
    if enc['count'] < min_occurs:
        # put in a separate category
        return -1
    else:
        return enc['code']


def group_data(data, degree, min_occurs):
    _, n = data.shape
    encoding = dict()
    for indexes in combinations(range(n), degree):
        for v in data[:, indexes]:
            encoding = dict_encode(encoding, tuple(v))
    new_data = []
    for indexes in combinations(range(n), degree):
        new_data.append([dict_decode(encoding, tuple(v), min_occurs) for v in data[:, indexes]])
    return np.array(new_data).T


def one_hot_encoder(data, keymap=None):
    """
    one_hot_encoder takes data matrix with categorical columns and
    converts it to a sparse binary matrix.
    Returns sparse binary matrix and keymap mapping categories to indices.
    If a keymap is supplied on input it will be used instead of creating one
    and any categories appearing in the data that are not in the keymap are ignored
    """

    if keymap is None:
        keymap = []
        for col in data.T:
            uniques = set(list(col))
            keymap.append(dict((key, i) for i, key in enumerate(uniques)))
    total_pts = data.shape[0]
    outdat = []
    for i, col in enumerate(data.T):
        km = keymap[i]
        num_labels = len(km)
        spmat = sparse.lil_matrix((total_pts, num_labels))
        for j, val in enumerate(col):
            if val in km:
                spmat[j, km[val]] = 1
        outdat.append(spmat)
    outdat = sparse.hstack(outdat).tocsr()
    return outdat, keymap


def create_test_submission(filename, urlids, predictions):
    content = ['urlid,label']
    for i, p in enumerate(predictions):
        content.append('%i,%f' % (urlids[i], p))
    f = open(filename, 'w')
    f.write('\n'.join(content))
    f.close()
    print('Saved')


def cv_loop(X, y, model, N, seed):
    mean_auc = 0.
    k_fold = KFold(len(y), N, indices=True, shuffle=True, random_state=seed)
    for train_ix, test_ix in k_fold:
        model.fit(X[train_ix], y[train_ix])
        preds = model.predict_proba(X[test_ix])[:, 1]
        auc = metrics.auc_score(y[test_ix], preds)
        #print("AUC (fold %d/%d): %f" % (i + 1, N, auc))
        mean_auc += auc
    return mean_auc / N


def main(train, test, submit, seed, min_occurs, good_features):
    start_time = time.time()
    print("Reading train dataset...")
    train_data = pd.read_csv(train, sep='\t')
    #print(train_data.head())

    print("Reading test dataset...")
    test_data = pd.read_csv(test, sep='\t')
    #print(test_data.head())

    y = np.array(train_data.label)
    all_data = pd.concat((train_data.ix[:, 1:-1], test_data.ix[:, 1:]), axis=0)

    num_train = np.shape(train_data)[0]

    # data["boilerplate_text"] = data["boilerplate"].map(
    #     lambda x: " ".join(filter(None,json.loads(x).values())))

    # convert boilerplate to bag of words for title, body and url
    for ii, cur in enumerate(('title', 'body', 'url')):
        curData = [json.loads(x)[cur] if cur in json.loads(x) else None for x in
                   all_data['boilerplate']]
        curData = ['empty' if x is None else x for x in curData]
        bow = feature_extraction.text.CountVectorizer(stop_words='english', min_df=10)
        bow = bow.fit_transform(curData)
        # TODO: gaat fout op "Reindexing only valid with uniquely valued Index objects"
        all_data = pd.concat((all_data, pd.DataFrame(bow.todense())), axis=1, ignore_index=True)
    all_data = all_data.drop(['boilerplate'])

    # clustering
    for clusters in (5, 10, 50):
        kproto = kmodes.KPrototypes(clusters)
        Xnum = all_data
        Xcat = all_data
        kproto.cluster([Xnum, Xcat], preRuns=5, prePctl=50, initMethod='Huang')
        all_data = pd.concat((all_data, kproto.clusters), axis=1, ignore_index=True)

    # Transform data
    print("Transforming data (%i instances)..." % num_train)
    # d_2 = group_data(all_data, degree=2, min_occurs=min_occurs)
    # d_3 = group_data(all_data, degree=3, min_occurs=min_occurs)
    # d_4 = group_data(all_data, degree=4, min_occurs=min_occurs)

    # X_train_all = np.hstack((all_data[:num_train], d_2[:num_train],
    #                          d_3[:num_train], d_4[:num_train]))
    # X_test_all  = np.hstack((all_data[num_train:], d_2[num_train:],
    #                          d_3[num_train:], d_4[num_train:]))
    X_train_all = all_data[:num_train]
    X_test_all = all_data[num_train:]

    num_features = X_train_all.shape[1]
    print("Total number of features %i" % num_features)

    rnd = random.Random()
    rnd.seed(seed * num_features)

    model = linear_model.LogisticRegression()
    model.C = 0.5 + rnd.random() * 3.5
    print("Logistic C parameter: %f" % model.C)

    # Xts holds one hot encodings for each individual feature in memory
    # speeding up feature selection
    Xts = [one_hot_encoder(X_train_all[:, [i]])[0] for i in range(num_features)]

    print("Performing aproximate greedy feature selection...")
    N = 10

    if good_features is None:
        score_hist = []
        good_features = set([])

        # Feature selection loop
        f_remain = list(range(len(Xts)))
        cur_best_score = -1
        cur_best_score_thres = 1.0
        while len(score_hist) < 2 or score_hist[-1][0] > score_hist[-2][0]:
            scores = []
            f_shuff = list(f_remain)
            rnd.shuffle(f_shuff)
            n_thres = 0.3679 * len(f_remain)
            i = 0
            iter_best_score = -1
            for f in f_shuff:

                i += 1
                feats = list(good_features) + [f]
                Xt = sparse.hstack([Xts[j] for j in feats]).tocsr()
                score = cv_loop(Xt, y, model, N, seed)
                if score < (cur_best_score * cur_best_score_thres):
                    f_remain.remove(f)
                    print("Discarded: %i (AUC = %f) " % (f, score))
                else:
                    scores.append((score, f))
                    if score > iter_best_score:
                        iter_best_score = score
                        if i > n_thres and iter_best_score > cur_best_score:
                            print("Early stop on iteration %i" % i)
                            break
            if len(scores) > 0:
                best_score = sorted(scores)[-1]
                f_remain.remove(best_score[1])
                if best_score[0] > cur_best_score:
                    good_features.add(best_score[1])
                    score_hist.append(best_score)
                    cur_best_score = best_score[0]
                print("Current features: %s (AUC = %f, remain = %i) " %
                      (list(good_features), best_score[0], len(f_remain)))
            else:
                break

    good_features = sorted(list(good_features))
    print("Selected features %s" % good_features)

    print("Performing hyperparameter selection...")
    # Hyperparameter selection loop
    Xt = sparse.hstack([Xts[j] for j in good_features]).tocsr()

    score_hist = []
    score = cv_loop(Xt, y, model, N, seed)
    score_hist.append((score, model.C))

    Cvals = np.logspace(-3, 4, 20, base=2)
    for C in Cvals:
        model.C = C
        score = cv_loop(Xt, y, model, N, seed)
        score_hist.append((score, C))
        print("C: %f Mean AUC: %f" % (C, score))
    model.C = sorted(score_hist)[-1][1]
    score = sorted(score_hist)[-1][0]
    print("Best (C, AUC): (%f, %f)" % (model.C, score))

    print("Performing One Hot Encoding on entire data set...")
    Xt = np.vstack((X_train_all[:, good_features], X_test_all[:, good_features]))
    Xt, keymap = one_hot_encoder(Xt)
    X_train = Xt[:num_train]
    X_test = Xt[num_train:]

    print("Training full model...")
    model.fit(X_train, y)

    print("Making prediction and saving results...")
    preds = model.predict_proba(X_test)[:, 1]
    create_test_submission(submit, preds)
    print("Total time %f minutes" % ((time.time() - start_time) / 60.0))


if __name__ == "__main__":

    name_def = '0.1'
    args = {'train': BASE_PATH + 'train.tsv',
            'test': BASE_PATH + 'test.tsv',
            'submit': BASE_PATH + name_def + 'submission.csv',
            'seed': 42,
            'min_occurs': 3,
            'good_features': []}

    print(args)
    main(**args)
