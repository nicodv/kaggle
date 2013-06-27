from sklearn import metrics, cross_validation, linear_model, decomposition, preprocessing, cluster
from scipy import sparse
import itertools

import numpy as np
import pandas as pd

DATA_DIR = '/home/nico/datasets/Kaggle/Employee/'
TRAIN_FILE = DATA_DIR+'train.csv'
TEST_FILE = DATA_DIR+'test.csv'
SUBM_FILE = DATA_DIR+'submission.csv'

SEED = 42

def group_data(data, degree=3, hash=hash):
    ''' 
    numpy.array -> numpy.array
    
    Groups all columns of data into all combinations of triples
    '''
    new_data = []
    m,n = data.shape
    for indices in itertools.combinations(range(n), degree):
        new_data.append([hash(tuple(v)) for v in data[:,indices]])
    return np.array(new_data).T

def create_submission(prediction):
    content = ['id,ACTION']
    for i, p in enumerate(prediction):
        content.append('%i,%f' %(i+1,p))
    f = open(SUBM_FILE, 'w')
    f.write('\n'.join(content))
    f.close()
    print 'Saved submission'

# This loop essentially from Paul Duan's starter code
def cv_loop(X, y, model, N):
    mean_auc = 0.
    for i in range(N):
        X_train, X_cv, y_train, y_cv = cross_validation.train_test_split(
                                       X, y, test_size=.20, 
                                       random_state = i*SEED)
        model.fit(X_train, y_train)
        preds = model.predict_proba(X_cv)[:,1]
        auc = metrics.auc_score(y_cv, preds)
        print "AUC (fold %d/%d): %f" % (i + 1, N, auc)
        mean_auc += auc
    return mean_auc/N


if __name__ == "__main__":
    
    print "Reading dataset..."
    train_data = pd.read_csv(TRAIN_FILE)
    test_data = pd.read_csv(TEST_FILE)
    
    # note: last column is duplicate, dropped here
    all_data = np.vstack((train_data.ix[:,1:-1], test_data.ix[:,1:-1]))
    
    num_train = np.shape(train_data)[0]
    
    # Transform data
    print "Transforming data..."
    dp = group_data(all_data, degree=2) 
    dt = group_data(all_data, degree=3)
    
    y = np.array(train_data.ACTION)
    X = all_data[:num_train]
    X_2 = dp[:num_train]
    X_3 = dt[:num_train]

    X_test = all_data[num_train:]
    X_test_2 = dp[num_train:]
    X_test_3 = dt[num_train:]

    X_train_all = np.hstack((X, X_2, X_3))
    X_test_all = np.hstack((X_test, X_test_2, X_test_3))
    num_features = X_train_all.shape[1]
    
    model = linear_model.LogisticRegression()
    
    # Xts holds one hot encodings for each individual feature in memory
    # speeding up feature selection 
    Xts = [preprocessing.OneHotEncoder(X_train_all[:,[i]])[0] for i in range(num_features)]
    
    print "Performing greedy feature selection..."
    score_hist = []
    N = 10
    good_features = set([])
    # Greedy feature selection loop
    while len(score_hist) < 2 or score_hist[-1][0] > score_hist[-2][0]:
        scores = []
        for f in range(len(Xts)):
            if f not in good_features:
                feats = list(good_features) + [f]
                Xt = sparse.hstack([Xts[j] for j in feats]).tocsr()
                score = cv_loop(Xt, y, model, N)
                scores.append((score, f))
                print "Feature: %i Mean AUC: %f" % (f, score)
        good_features.add(max(scores)[1])
        score_hist.append(max(scores))
        print "Current features: %s" % sorted(list(good_features))
    
    # Remove last added feature from good_features
    good_features.remove(score_hist[-1][1])
    good_features = sorted(list(good_features))
    print "Selected features %s" % good_features
    # good_features= [0, 7, 8, 10, 29, 36, 37, 42, 47, 53, 55, 64, 65, 66, 67, 69, 71, 79, 81, 82]
    
    print "Performing hyperparameter selection..."
    # Hyperparameter selection loop
    score_hist = []
    Xt = sparse.hstack([Xts[j] for j in good_features]).tocsr()
    Cvals = np.logspace(-4, 4, 20, base=2)
    for C in Cvals:
        model.C = C
        score = cv_loop(Xt, y, model, N)
        score_hist.append((score,C))
        print "C: %f Mean AUC: %f" %(C, score)
    bestC = max(score_hist)[1]
    model.C = bestC
    print "Best C value: %f" % (bestC)
    
    model.C = 2.5
    
    print "Performing One Hot Encoding on entire dataset..."
    Xt = np.vstack((X_train_all[:,good_features], X_test_all[:,good_features]))
    Xt, keymap = preprocessing.OneHotEncoder(Xt)
    X_train = Xt[:num_train]
    X_test = Xt[num_train:]
    
    print "Training full model..."
    model.fit(X_train, y)
    
    print "Making prediction and saving results..."
    preds = model.predict_proba(X_test)[:,1]
    create_submission(preds)
    
