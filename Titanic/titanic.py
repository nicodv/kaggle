import numpy as np
import pandas as pd
import re
import itertools
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
	_, nfeat = data.shape
	for indices in itertools.combinations(range(n_feat), degree):
		new_data.append(data[:,indices])
	return preprocessing.LabelEncoder().fit_transform(new_data)

def prepare_data(d):
	
	# delete unused columns
	d = d.drop(['ticket'],axis=1)
	
	# set missing embarks to 'S', by far the most common value. couldn't find any other clues
	d.embarked[d.embarked.isnull()] = 'S'
	
	# one-hot encoding for categorical features
	d, _, _ = one_hot_dataframe(d, ['sex','embarked'], replace=True)
	
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
	f = lambda x: x.fillna(x.median())
	d.fare = fares.transform(f)
	# now rank the fares
	d['farerank'] = d.fare.rank() / len(d.fare)
	
	# create some new combined features
	d['pvar'] = d.parch+1 * d.sibsp+1
	
	combo = ['sex','pclass','parch','sibsp','cabin','embarked']
	comb2feats = construct_combined_features(d[combo].as_array(), degree=2)
	d = pd.concat(d, comb2feats)
	
	return d

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
	cv = cross_validation.StratifiedKFold(targets, n_folds=8)
	
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
	
	num_train = len(traindata)
	
	totdata = pd.concat(traindata, testdata)
	totdata = prepare_data(totdata)
	
	# feature selection
	selector = feature_selection.SelectKBest(f_classif, k=25)
	traindata = totdata.ix[:,:num_train]
	selector.fit_transform(traindata, targets)
	testdata = selector.transform(totdata.ix[:,num_train:])
	
	models = train_model(traindata, targets)
	
	# make prediction
	prediction = [[0]*418 for i in range(len(models))]
	for i in range(len(models)):
		prediction[i] = models[i].predict(testdata[colnames[:-1]]).tolist()
	
	pred = pd.DataFrame(prediction).median().astype(int)
	
	# save to CSV
	pred.to_csv(DATA_DIR+'submission.csv', sep=',', header=False, index=False)
	
