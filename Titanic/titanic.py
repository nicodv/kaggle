import numpy as np
import pandas as pd
import re
from sklearn import cross_validation, ensemble, pipeline, \
            grid_search, metrics, svm, preprocessing, neighbors, decomposition

DATA_DIR = '/home/nico/datasets/Kaggle/Titanic/'

def families(train, test):
    trainr = train[['name','parch','sibsp','survived']]
    trainr['ind'] = trainr.index
    testr = test[['name','parch','sibsp']]
    testr['ind'] = testr.index + 10000
    tot = pd.concat([trainr, testr]).reset_index()
    tot['fvar'] = tot.parch + tot.sibsp
    tot = tot.drop(['parch','sibsp'], axis=1)
    # find family name (at start, everything before comma)
    fnre = re.compile('^(.+?),')
    tot['famname'] = tot.name.map(lambda x: fnre.search(x).group()[:-1])
    # survival chance = mean of rest of family
    survived = tot.groupby(['famname','fvar']).survived
    f = lambda x: x.fillna(x.mean())
    tot['famscore'] = survived.transform(f)
    # wait, loners shouldn't look at other loners
    tot.famscore[tot.fvar==0] = np.nan
    tot.famscore = tot.famscore.fillna(0.5)
    
    train['famscore'] = pd.merge(trainr,tot[['ind','famscore']], how='left', on='ind')['famscore']
    test['famscore'] = pd.merge(testr,tot[['ind','famscore']], how='left', on='ind')['famscore']
    return train, test

def prepare_data(d):
    
    # delete unused columns
    d = d.drop(['ticket'],axis=1)
    
    d.sex[d.sex=='female'] = -1
    d.sex[d.sex=='male'] = 1
    
    d['embarked1'] = 0
    d['embarked2'] = 0
    d['embarked3'] = 0
    d.embarked1[d.embarked=='C'] = 1
    d.embarked2[d.embarked=='S'] = 1
    d.embarked3[d.embarked=='Q'] = 1
    # set missings to 'S', by far the most common value. couldn't find any other clues
    d.embarked2[d.embarked.isnull()] = 1
    
    # categories based on title in name
    re1 = re.compile("Mr.|Dr.|Col.|Major.|Rev.")
    re2 = re.compile("Mrs.|Mlle.|Don.|Countess.|Jonkheer.")
    re3 = re.compile("Miss.")
    re4 = re.compile("Master.")
    d['title1'] = d.name.map(lambda x: int(bool(re1.search(x))))
    d['title2'] = d.name.map(lambda x: int(bool(re2.search(x))))
    d['title3'] = d.name.map(lambda x: int(bool(re3.search(x))))
    d['title4'] = d.name.map(lambda x: int(bool(re4.search(x))))
    
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
    d = d.drop(['embarked'],axis=1)
    f = lambda x: x.fillna(x.median())
    d.fare = fares.transform(f)
    
    # now rank the fares
    d['farerank'] = d.fare.rank() / len(d.fare)
    
    # create some new combined features
    d['pvar'] = d.parch+1 * d.sibsp+1
    
    return d

traindata = pd.read_csv(DATA_DIR+'train.csv')
testdata = pd.read_csv(DATA_DIR+'test.csv')

traindata, testdata = families(traindata, testdata)

traindata = prepare_data(traindata)
testdata = prepare_data(testdata)

# SELECT INPUT VARIABLES HERE
colnames = ['pclass','farerank','age','pvar','famscore','survived']

traindata = traindata[colnames]
testdata = testdata[colnames[:-1]]

def train_model(traindata, targets):
    
    models = [ \
    ensemble.GradientBoostingClassifier(n_estimators=50, learning_rate=0.2, \
    max_depth=1, subsample=0.4, max_features=4, min_samples_leaf=10), \
    ensemble.GradientBoostingClassifier(n_estimators=100, learning_rate=0.2, \
    max_depth=2, subsample=0.5, max_features=4, min_samples_leaf=10), \
    ensemble.GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, \
    max_depth=3, subsample=0.75, max_features=3, min_samples_leaf=20), \
    ensemble.GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, \
    max_depth=4, subsample=0.9, max_features=2, min_samples_leaf=30), \
    ensemble.GradientBoostingClassifier(n_estimators=400, learning_rate=0.05, \
    max_depth=5, subsample=0.95, max_features=2, min_samples_leaf=30) \
    ]
    
    # use StratifiedKFold, because survived 0/1 is not evenly distributed
    cv = cross_validation.StratifiedKFold(targets, n_folds=5)
    
    scores = [0]*len(models)
    for i in range(len(models)):
        # get scores
        scores[i] = cross_validation.cross_val_score(models[i], traindata, targets, \
                cv=cv, n_jobs=-1, score_func=metrics.accuracy_score)
        print("Cross-validation accuracy on the training set for model %d:" % i)
        print("%0.3f (+/-%0.03f)" % (scores[i].mean(), scores[i].std() / 2))
        
        models[i].fit(traindata, targets)
        
    return models

# preprocessing
def preproc(d):
    meand = d.mean()
    stdd = d.std()
    d = d.map(lambda x: (x - meand) / stdd)

    return d, meand, stdd

for i in colnames:
    if i in ('pvar','age','farerank','famscore'):
        traindata[i], meand, stdd = preproc(traindata[i])
        testdata[i] = testdata[i].map(lambda x: (x - meand) / stdd)

models = train_model(traindata[colnames[:-1]], targets)

# make prediction
prediction = [[0]*418 for i in range(len(models))]
for i in range(len(models)):
    prediction[i] = models[i].predict(testdata[colnames[:-1]]).tolist()

pred = pd.DataFrame(prediction).median().astype(int)

# save to CSV
pred.to_csv(DATA_DIR+'submission.csv', sep=',', header=False, index=False)
