import numpy as np
import pandas as pd
import re, string
import itertools
from string import ascii_lowercase
from sklearn import feature_extraction, feature_selection, preprocessing, pipeline, \
                    grid_search, cross_validation, ensemble, metrics

DATA_DIR = '/home/nico/datasets/Kaggle/Titanic/'

def one_hot_dataframe(data, cols, replace=False):
    '''Takes a dataframe and a list of columns that need to be encoded.
    Returns a 3-tuple comprising the data, the vectorized data,
    and the fitted vectorizor.
    '''
    vec = feature_extraction.DictVectorizer()
    mkdict = lambda row: dict((col, row[col]) for col in cols)
    vecData = pd.DataFrame(vec.fit_transform(data[cols].apply(mkdict, axis=1)).toarray())
    vecData.columns = vec.get_feature_names()
    vecData.index = data.index
    if replace is True:
        data = data.drop(cols, axis=1)
        data = data.join(vecData)
    return (data, vecData, vec)

def construct_combined_features(data, degree=2):
    new_data = []
    _,nfeat = data.shape
    for indices in itertools.combinations(range(nfeat), degree):
        new_data.append([str(hash(tuple(v))) for v in data[:,indices]])
    return np.array(new_data).T

def prepare_data(d):

    # delete unused columns
    d = d.drop(['ticket'],axis=1)
    
    # set missing embarks to 'S', by far the most common value. couldn't find any other clues
    d.embarked[d.embarked.isnull()] = 'S'
    
    d.cabin[~d.cabin.isnull()] = d.cabin[~d.cabin.isnull()].map(lambda x: x[0])
    # E and D appear to be safe areas
    d.cabin[d.cabin=='E'] = 1
    d.cabin[d.cabin=='D'] = 2
    d.cabin[d.cabin=='A'] = 3
    d.cabin[d.cabin=='B'] = 3
    d.cabin[d.cabin=='T'] = 3
    d.cabin[d.cabin=='C'] = 4
    d.cabin[d.cabin=='F'] = 5
    d.cabin[d.cabin=='G'] = 5
    
    cabins = d.groupby(['pclass']).cabin
    # fill with median of these groups
    f = lambda x: x.fillna(round(x.median()))
    d.cabin = cabins.transform(f)
    
    # convert nominal data to integers
    # simple encoding of first 3 characters of name
    d.name = d.name.map(lambda x: x[:3])
    d.name = d.name.map(lambda x: x.replace("'",'z'))
    d.name = d.name.map(lambda x: string.lowercase.index(x[0].lower())*900 + \
    string.lowercase.index(x[1].lower())*30 + string.lowercase.index(x[1].lower()))
    
    # traveling in what sort of group? used for age
    d['member'] = np.nan
    # traveling alone
    d.member[(d.sibsp == 0) & (d.parch == 0)] = 1
    # couple or grown-up siblings
    d.member[(d.sibsp > 0) & (d.parch == 0)] = 2
    # child member
    d.member[((d.sibsp == 2) & (d.parch > 0)) | (d.sibsp > 2)] = 3
    # parent member
    d.member[((d.sibsp > 0) & (d.parch > 1)) | ((d.sibsp == 0) & (d.parch > 1))] = 4
    # other
    d.member[((d.sibsp < 2) & (d.parch == 1))] = 5
    
    # fill missing values
    ages = d.groupby(['sex','member']).age
    # fill with median of these groups
    f = lambda x: x.fillna(x.median())
    d.age = ages.transform(f)
    
    # median price per class (above threshold for lucky cheapos)
    fares = d.groupby(['pclass','embarked']).fare
    f = lambda x: x.fillna(x.median())
    d.fare = fares.transform(f)
    # now rank the fares
    d['farerank'] = d.fare.rank() / len(d.fare)
    
    # create some new combined features
    d['pvar'] = d.parch+1 * d.sibsp+1
    
    combo = ['sex','pclass','parch','sibsp','cabin','embarked']
    L = list(ascii_lowercase)
    LL = [2*x for x in L]
    LLL = [3*x for x in L]
    comb2feats = pd.DataFrame(construct_combined_features(np.array(d[combo]), degree=2), columns=L[:15], dtype=object)
    comb3feats = pd.DataFrame(construct_combined_features(np.array(d[combo]), degree=3), columns=LL[:20], dtype=object)
    comb4feats = pd.DataFrame(construct_combined_features(np.array(d[combo]), degree=4), columns=LLL[:15], dtype=object)
    d = pd.concat([d, comb2feats, comb3feats, comb4feats], axis=1)
    
    # one-hot encoding for categorical features
    d, _, _ = one_hot_dataframe(d, ['embarked', 'cabin'], replace=True)
    d, _, _ = one_hot_dataframe(d, L[:15], replace=True)
    d, _, _ = one_hot_dataframe(d, LL[:20], replace=True)
    d, _, _ = one_hot_dataframe(d, LLL[:15], replace=True)
    
    return d

def train_model(traindata, targets):
    
    models = [ \
#    ensemble.GradientBoostingClassifier(n_estimators=250, learning_rate=0.1, \
#    max_depth=4, subsample=0.3, max_features=8, min_samples_leaf=10), \
#    ensemble.GradientBoostingClassifier(n_estimators=500, learning_rate=0.06, \
#    max_depth=6, subsample=0.5, max_features=6, min_samples_leaf=15), \
#    ensemble.GradientBoostingClassifier(n_estimators=750, learning_rate=0.02, \
#    max_depth=7, subsample=0.7, max_features=4, min_samples_leaf=20) \
    ensemble.RandomForestClassifier(n_estimators=250, \
    max_depth=2, max_features='sqrt', min_samples_leaf=25), \
    ensemble.RandomForestClassifier(n_estimators=500, \
    max_depth=3, max_features='sqrt', min_samples_leaf=30), \
    ensemble.RandomForestClassifier(n_estimators=750, \
    max_depth=2, max_features='sqrt', min_samples_leaf=25) \
    ]
    
    # use StratifiedKFold, because survived 0/1 is not evenly distributed
    cv = cross_validation.StratifiedKFold(targets, n_folds=3)
    
    scores = [0]*len(models)
    for i in range(len(models)):
        # get scores
        scores[i] = cross_validation.cross_val_score(models[i], traindata, targets, \
                    cv=cv, n_jobs=-1, score_func=metrics.accuracy_score)
        print("Cross-validation accuracy on the training set for model %d:" % i)
        print("%0.3f (+/-%0.03f)" % (scores[i].mean(), scores[i].std() / 2))

        models[i].fit(traindata, targets)

    return models

if __name__ == '__main__':
    traindata = pd.read_csv(DATA_DIR+'train.csv')
    testdata = pd.read_csv(DATA_DIR+'test.csv')
    targets = traindata['survived']
    #traindata = traindata.drop(['survived'],axis=1)
    
    num_train = len(traindata)
    
    totdata = pd.concat([traindata, testdata], ignore_index=True)
    totdata = prepare_data(totdata)
    
    traindata = totdata.ix[:num_train-1,:]
    testdata = totdata.ix[num_train:,:]
    
    # separate male/female data
    Fmask = (traindata.sex=='female')
    traindataF = traindata.drop(Fmask[Fmask==False].index)
    traindataM = traindata.drop(Fmask[Fmask==True].index)
    traindataF.reset_index(inplace=True)
    traindataM.reset_index(inplace=True)
    del traindataF['sex']
    del traindataM['sex']
    targetsF = traindataF.survived
    targetsM = traindataM.survived
    del traindataF['survived']
    del traindataM['survived']
    
    Fmask = (testdata.sex=='female')
    testdataF = testdata.drop(Fmask[Fmask==False].index)
    testdataM = testdata.drop(Fmask[Fmask==True].index)
    testdataM.reset_index(inplace=True)
    traindataM.reset_index(inplace=True)
    del testdataF['sex']
    del testdataM['sex']
    
    # feature selection
    selectorF = feature_selection.SelectKBest(feature_selection.f_classif, k=20)
    traindataF = selectorF.fit_transform(traindataF, targetsF)
    testdataF = selectorF.transform(testdataF.fillna(value=0))
    selectorM = feature_selection.SelectKBest(feature_selection.f_classif, k=20)
    traindataM = selectorM.fit_transform(traindataM, targetsM)
    testdataM = selectorM.transform(testdataM.fillna(value=0))
    
    modelsF = train_model(traindataF, targetsF)
    modelsM = train_model(traindataM, targetsM)
    
    # make female prediction
    prediction = [[0]*418 for i in range(len(modelsF))]
    for i in range(len(modelsF)):
        prediction[i] = modelsF[i].predict(testdataF).tolist()
        
    predF = pd.DataFrame(prediction).median().astype(int)
    
    predtot = [0]*418
    predtot = pd.Series(predtot)
    
    Finds = [x-891 for x in Fmask[Fmask==True].index.tolist()]
    predtot[Finds] = predF
    
    # make male prediction
    prediction = [[0]*418 for i in range(len(modelsM))]
    for i in range(len(modelsM)):
        prediction[i] = modelsM[i].predict(testdataM).tolist()
        
    predM = pd.DataFrame(prediction).median().astype(int)
    
    Minds = [x-891 for x in Fmask[Fmask==False].index.tolist()]
    predtot[Minds] = predM
    
    pid = pd.Series(range(892,1310))
    predfinal = pd.DataFrame({'PassengerID': pid, 'Survived': predtot})
    
    # save to CSV
    predfinal.to_csv(DATA_DIR+'submission.csv', sep=',', header=True, index=False)
    
