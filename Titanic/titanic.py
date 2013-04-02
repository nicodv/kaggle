import numpy as np
import pandas as pd
import string
from sklearn import cross_validation, ensemble, pipeline, \
            grid_search, metrics, svm, preprocessing, neighbors, decomposition

DATA_DIR = '/home/nico/datasets/Kaggle/Titanic/'

traindata = pd.read_csv(DATA_DIR+'train.csv')
testdata = pd.read_csv(DATA_DIR+'test.csv')

def prepare_data(d):
    
    # delete unused columns
    d = d.drop(['ticket'],axis=1)
    
    # convert nominal data to integers
    # simple encoding of first 3 characters of name
    d.name = d.name.map(lambda x: x[:3])
    d.name = d.name.map(lambda x: x.replace("'",'z'))
    d.name = d.name.map(lambda x: string.lowercase.index(x[0].lower())*900 + \
        string.lowercase.index(x[1].lower())*30 + string.lowercase.index(x[1].lower()))
    
    d.sex[d.sex=='female'] = -1
    d.sex[d.sex=='male'] = 1
    
    d.embarked[d.embarked=='C'] = 1
    d.embarked[d.embarked=='S'] = 2
    d.embarked[d.embarked=='Q'] = 3
    
    d.cabin[-d.cabin.isnull()] = d.cabin[-d.cabin.isnull()].map(lambda x: x[0])
    # D and E appear to be safe areas
    d.cabin[d.cabin=='E'] = 1
    d.cabin[d.cabin=='D'] = 2
    d.cabin[d.cabin=='A'] = 3
    d.cabin[d.cabin=='B'] = 3
    d.cabin[d.cabin=='C'] = 4
    d.cabin[d.cabin=='F'] = 5
    d.cabin[d.cabin=='G'] = 5
    d.cabin[d.cabin=='T'] = 3
    
    cabins = d.groupby(['pclass']).cabin
    # fill with median of these groups
    f = lambda x: x.fillna(round(x.median()))
    d.cabin = cabins.transform(f)
    
    # 2 values are missing. set to 2, by far the most common value
    d.embarked[d.embarked.isnull()] = 2
    
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
    
    d['companions'] = np.log(d.sibsp + d.parch + 1)
    
    # fill missing values
    ages = d.groupby(['sex','member']).age
    # fill with median of these groups
    f = lambda x: x.fillna(x.median())
    d.age = ages.transform(f)
    
    # fill that one missing value
    d.fare = d.fare.fillna(0)
    # median price per class (above threshold for lucky cheapos)
    fares = d.groupby(['pclass','embarked']).fare
    f = lambda x: x.fillna(x.median()) / x.median()
    d.fare = fares.transform(f)
    
    # now rank the fares
    d['farerank'] = d.fare.rank() / len(d.fare)
    d['status'] = d.pclass - d.farerank
    
    # create some new combined features
    d['sexage'] = d.sex * d.age
    d['pvar'] = d.parch+1 * d.sibsp+1
    d['agepvar'] = d.age / np.sqrt(d.pvar)
    
    return d

traindata = prepare_data(traindata)
testdata = prepare_data(testdata)
totdata = traindata
totdata['train'] = 1
tempdata = testdata
tempdata['train'] = 0
totdata = totdata.append(tempdata)

def select_columns(d, colnames):
    # select columns
    d = d[colnames]
    
    return d

# SELECT INPUT VARIABLES HERE
colnames = ['sex','name','pclass','farerank','age','pvar','survived']

traindata = select_columns(traindata, colnames)
testdata = select_columns(testdata, colnames[:-1])

def train_model(traindata, targets):
    
    models = [ \
    neighbors.KNeighborsClassifier(n_neighbors=40, weights='distance', algorithm='brute', warn_on_equidistant=False), \
    #svm.SVC(C=100, kernel='rbf', class_weight='auto'), \
    ensemble.RandomForestClassifier(n_estimators=100, max_features='log2', \
    compute_importances=True, oob_score=False, min_samples_split=20, criterion='entropy'), \
    ensemble.GradientBoostingClassifier(n_estimators=100, learning_rate=0.2, \
    max_depth=1, subsample=0.5, max_features=3, min_samples_leaf=10), \
    ensemble.GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, \
    max_depth=2, subsample=0.75, max_features=3, min_samples_leaf=20), \
    ensemble.GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, \
    max_depth=3, subsample=0.95, max_features=3, min_samples_leaf=30) \
    ]
    
    # use StratifiedKFold, because survived 0/1 is not evenly distributed
    cv = cross_validation.StratifiedKFold(targets, n_folds=10)
    
    # pipelining
    #estimators = [('reduce_dim', PCA()), ('svm', SVC())]
    #clf = pipeline.Pipeline(estimators)
    
    # access with '<estimator>__<parameter>'
    #params = dict(reduce_dim__n_components=[2, 5, 10], svm__C=[0.1, 10, 100])
    #gridSearch = grid_search.GridSearchCV(model, param_grid=params)
    
    scores = [0]*len(models)
    for i in range(len(models)):
        # get scores
        scores[i] = cross_validation.cross_val_score(models[i], traindata, targets, \
                cv=cv, n_jobs=-1, score_func=metrics.accuracy_score)
        print "Cross-validation accuracy on the training set for model %d:" % i
        print "%0.3f (+/-%0.03f)" % (scores[i].mean(), scores[i].std() / 2)
        
        models[i].fit(traindata, targets)
        
    return models


Fmask = (traindata.sex==-1)
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

Fmask = (testdata.sex==-1)
testdataF = testdata.drop(Fmask[Fmask==False].index)
testdataM = testdata.drop(Fmask[Fmask==True].index)
testdataM.reset_index(inplace=True)
traindataM.reset_index(inplace=True)
del testdataF['sex']
del testdataM['sex']

# preprocessing
def preproc(d):
    meand = d.mean()
    stdd = d.std()
    d = d.map(lambda x: (x - meand) / stdd)

    return d, meand, stdd

for i in colnames:
    if (i != 'sex') & (i != 'survived'):
        traindataF[i], meand, stdd = preproc(traindataF[i])
        testdataF[i] = testdataF[i].map(lambda x: (x - meand) / stdd)
        traindataM[i], meand, stdd = preproc(traindataM[i])
        testdataM[i] = testdataM[i].map(lambda x: (x - meand) / stdd)

modelsF = train_model(traindataF[colnames[1:-1]], targetsF)
modelsM = train_model(traindataM[colnames[1:-1]], targetsM)

# make prediction
prediction = [[0]*418 for i in range(len(modelsF))]
for i in range(len(modelsF)):
    prediction[i] = modelsF[i].predict(testdataF[colnames[1:-1]]).tolist()

predF = pd.DataFrame(prediction).median().astype(int)

predtot = [0]*418
predtot = pd.Series(predtot)

predtot[Fmask[Fmask==True].index.tolist()] = predF

prediction = [[0]*418 for i in range(len(modelsM))]
for i in range(len(modelsM)):
    prediction[i] = modelsM[i].predict(testdataM[colnames[1:-1]]).tolist()

predM = pd.DataFrame(prediction).median().astype(int)

predtot[Fmask[Fmask==False].index.tolist()] = predM

# export
totdata.survived[totdata.train==0] = predtot
totdata.to_csv(DATA_DIR+'rawdata.txt', sep=',', header=True, index=False)

print "Fraction survivors: %d / %d" % (sum(predtot), len(predtot))

# save to CSV
predtot.to_csv(DATA_DIR+'submission.csv', sep=',', header=False, index=False)
