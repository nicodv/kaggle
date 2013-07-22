import numpy as np
import pandas as pd
from multiprocessing import Pool
import itertools

from sklearn import metrics, cross_validation, linear_model, decomposition, preprocessing, cluster

DATA_DIR = '/home/nico/datasets/Kaggle/Employee/'
TRAIN_FILE = DATA_DIR+'train.csv'
TEST_FILE = DATA_DIR+'test.csv'
SUBM_FILE = DATA_DIR+'submission.csv'

SEED = 42

def construct_combined_features(data, degree=2):
    '''Combine features into a set of new features that express the
double/triple/etc. combinations of original features.
'''
    
    new_data = []
    _, nfeat = data.shape
    for indices in itertools.combinations(range(n_feat), degree):
        new_data.append(data[:,indices])
    
    return preprocessing.LabelEncoder().fit_transform(new_data)

def greedy_feature_selection(all_features, training_data):
    pool = Pool(processes=4)

    best_features = []
    last_score = 0

    while True:
        test_feature_sets = [[f] + best_features
                             for f in all_features
                             if f not in best_features]

        args = [(feature_set, training_data.ACTION)
                for feature_set in test_feature_sets]

        scores = pool.map(features_score, args)

        (score, feature_set) = max(zip(scores, test_feature_sets))
        print feature_set
        print score
        if score <= last_score:
            break
        last_score = score
        best_features = feature_set

    pool.close()
    return best_features

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
    
    # Change the prediction function for LogisticRegression
    new_predict = lambda M, x: M.predict_proba[:,1]
    model.predict = new_predict
    cv = cross_validation.StratifiedShuffleSplit(y, n_iter=N)
    
    # replacement for cv_loop
    scores = cross_validation.cross_val_score(model, X, y, score_func=metrics.auc_score, n_jobs=-1, cv=cv)
    
    return no.mean(scores)
    

if __name__ == "__main__":
    
    print "Reading dataset..."
    train_data = pd.read_csv(TRAIN_FILE)
    test_data = pd.read_csv(TEST_FILE)
    
    # note: last column is duplicate, so dropped here
    all_data = np.vstack((train_data.ix[:,1:-1], test_data.ix[:,1:-1]))
    
    num_train = np.shape(train_data)[0]
    
    # Transform data
    print "Transforming data..."
    dp = construct_combined_features(all_data, degree=2)
    dt = construct_combined_features(all_data, degree=3)
    
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
    
