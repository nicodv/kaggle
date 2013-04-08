from dateutil.parser import parse
from datetime import datetime
import pandas as pd
import os
import numpy as np
from collections import defaultdict
from sklearn import ensemble, metrics, cross_validation, svm
import pylab

data_path = '/home/nico/temp/Kaggle data/Bulldozers/'
submission_path = '/home/nico/Dropbox/Research Projects/Kaggle/Bulldozers/'

def get_train_df(data_path):
    train = pd.read_csv(os.path.join(data_path, "Train.csv"),
        converters={"saledate": parse})
    train = train.drop(['ModelID','YearMade','fiModelDesc','fiBaseModel', \
        'fiSecondaryDesc','fiModelSeries','fiModelDescriptor','fiProductClassDesc','ProductGroup', \
        'ProductGroupDesc'], axis=1)
    app = pd.read_csv(os.path.join(data_path, "Machine_Appendix.csv"),
        converters={"saledate": parse})
    train = pd.merge(train, app, on='MachineID', how='left')
    return train

def get_test_df(data_path):
    test = pd.read_csv(os.path.join(data_path, "Valid.csv"),
        converters={"saledate": parse})
    test = test.drop(['ModelID','YearMade','fiModelDesc','fiBaseModel', \
        'fiSecondaryDesc','fiModelSeries','fiModelDescriptor','fiProductClassDesc','ProductGroup', \
        'ProductGroupDesc'], axis=1)
    app = pd.read_csv(os.path.join(data_path, "Machine_Appendix.csv"),
        converters={"saledate": parse})
    test = pd.merge(test, app, on='MachineID', how='left')
    return test

def get_train_test_df(data_path):
    return get_train_df(data_path), get_test_df(data_path)

def write_submission(submission_name, predictions, data_path, submission_path):
    test = get_test_df(data_path)
    test = test.join(pd.DataFrame({"SalePrice": predictions}))
    
    test[["SalesID", "SalePrice"]].to_csv(os.path.join(submission_path,
        submission_name), index=False)

train, test = get_train_test_df(data_path)

columns = set(train.columns)
columns.remove("SalesID")
columns.remove("saledate")
columns.remove("SalePrice")

traindata = pd.DataFrame(train["saledate"])
testdata = pd.DataFrame(test["saledate"])

for col in columns:
    dropcols = ['MachineID','Turbocharged','Grouser_Tracks','Pad_Type','Backhoe_Mounting','Grouser_Type', \
                'UsageBand','Blade_Type','Stick','auctioneerID','datasource','Forks','Pattern_Changer', \
                'Differential_Type','Thumb','Hydraulics_Flow','Transmission','Steering_Controls', \
                'Track_Type','Undercarriage_Pad_Width','Coupler_System','Travel_Controls', \
                'Ripper','Coupler','Hydraulics','fiModelSeries','ProductGroupDesc', \
                'Tip_Control','Scarifier','Blade_Extension','Drive_System','fiModelDesc','Stick_Length', \
                'PrimaryLower','fiProductClassDesc','fiManufacturerDesc','Enclosure_Type','Pushblock']
    if col not in dropcols:
        types = set(type(x) for x in train[col])
        if str in types:
            s = set(x for x in train[col])
            str_to_categorical = defaultdict(lambda: -1, [(x[1], x[0]) for x in enumerate(s)])
            traindata = traindata.join(pd.DataFrame({col: [str_to_categorical[x] for x in train[col]]}, index=train.index))
            testdata = testdata.join(pd.DataFrame({col: [str_to_categorical[x] for x in test[col]]}, index=test.index))
        else:
            traindata = traindata.join(train[col])
            testdata = testdata.join(test[col])

traindata['age'] = traindata.saledate.map(lambda x: x.year) - traindata.MfgYear
testdata['age'] = testdata.saledate.map(lambda x: x.year) - testdata.MfgYear
traindata = traindata.drop(['MfgYear'], axis=1)
testdata = testdata.drop(['MfgYear'], axis=1)

# fill ridiculous age values
def fillcond_age(group):
    mask = ((group > 100) | group.isnull())
    group[mask] = group[~mask].median()
    return group

ages = traindata.groupby('fiBaseModel').age
traindata.age = ages.transform(fillcond_age)
ages = traindata.groupby('ProductGroup').age
traindata.age = ages.transform(fillcond_age)
traindata.age[traindata.age.isnull()] = np.median(traindata.age)

ages = testdata.groupby('fiBaseModel').age
testdata.age = ages.transform(fillcond_age)
ages = testdata.groupby('ProductGroup').age
testdata.age = ages.transform(fillcond_age)
testdata.age[testdata.age.isnull()] = np.median(testdata.age)

traindata.PrimaryUpper[traindata.PrimaryUpper.isnull()] = np.median(traindata.PrimaryUpper)
testdata.PrimaryUpper[testdata.PrimaryUpper.isnull()] = np.median(testdata.PrimaryUpper)

traindata.MachineHoursCurrentMeter[traindata.MachineHoursCurrentMeter.isnull()] = 0
testdata.MachineHoursCurrentMeter[testdata.MachineHoursCurrentMeter.isnull()] = 0

#==============================================================================
# PRICE MODEL
#==============================================================================
targets = np.log(train["SalePrice"])
inputs = traindata.drop(['saledate'],axis=1)

pricemodel = ensemble.RandomForestRegressor(n_estimators=200, n_jobs=1, compute_importances = True, \
            max_features='sqrt', min_samples_split=40, max_depth=5)
#pricemodel = ensemble.GradientBoostingRegressor(learning_rate=0.1, n_estimators=500, \
#                subsample=0.5, max_features=5)

cv = cross_validation.KFold(len(targets))

scores = cross_validation.cross_val_score(pricemodel, inputs, targets, \
            cv=cv, n_jobs=-1, score_func=metrics.mean_squared_error)
print "Cross-validation accuracy on the training set:"
print "%0.3f (+/-%0.03f)" % (scores.mean(), scores.std() / 2)

pricemodel.fit(inputs, targets)

imp = sorted(zip(traindata.columns, pricemodel.feature_importances_), key=lambda tup: tup[1], reverse=True)
for fea in imp:
    print(fea)

traindata['estPrice'] = pricemodel.predict(inputs)
traindata['estFactor'] = np.log(train["SalePrice"]) / traindata.estPrice

#==============================================================================
# ECONOMIC MODEL
#==============================================================================
# make 2 matching time series
estFactors = pd.DataFrame(traindata.groupby(['saledate']).estFactor.mean())

# read from .CSV and shift in time to align properly with estFactors
ext1Data = pd.read_csv(data_path+'TTLCONS.csv', names=['saledate','econ1'], skiprows=1)
ext1Data.saledate = ext1Data.saledate.map(lambda x: datetime.strptime(x, '%Y-%m-%d').date())
ext1Data = ext1Data.set_index('saledate').shift(3)
ext2Data = pd.read_csv(data_path+'HTRUCKSSA.csv', names=['saledate','econ2'], skiprows=1)
ext2Data.saledate = ext2Data.saledate.map(lambda x: datetime.strptime(x, '%Y-%m-%d').date())
ext2Data = ext2Data.set_index('saledate').shift(3)
ext3Data = pd.read_csv(data_path+'CUSR0000SETA02.csv', names=['saledate','econ3'], skiprows=1)
ext3Data.saledate = ext3Data.saledate.map(lambda x: datetime.strptime(x, '%Y-%m-%d').date())
ext3Data = ext3Data.set_index('saledate').shift(2)
extData = pd.merge(ext1Data,pd.merge(ext2Data,ext3Data,left_index=True,right_index=True,how='outer'),left_index=True,right_index=True,how='outer')

# smooth the stuff out a bit
extData = pd.rolling_mean(extData, 6)

# training data, exponentially smoothed
totData = extData.diff().join(estFactors,how='inner')
totData = pd.ewma(totData, span=12)

# use SVM regression for economic model
# tapped delayed lines so that we are actually forecasting, not cheating
inputsd = totData[['econ1','econ2','econ3']]
inputs = pd.concat([inputsd.shift(1), inputsd.shift(2), inputsd.shift(3), inputsd.shift(4), \
                inputsd.shift(7), inputsd.shift(13), inputsd.shift(25)], axis=1).fillna(0)
targets = totData.estFactor

# normalize
inputs = (inputs - inputs.mean()) / inputs.std()

targmean = targets.mean()
targstd = targets.std()
targets = (targets - targmean) / targstd

econ = svm.SVR(kernel='rbf', degree=2, gamma=0.0, C=0.25, epsilon=0.2, shrinking=True)
econ = econ.fit(inputs, targets)
print "Economic model fit: %f" % econ.score(inputs, targets)
# plot econ model
pylab.plot(targets, color='black')
pylab.plot(econ.predict(inputs), color='red')
pylab.show()

trainpreds = econ.predict(inputs)
# shift 1 in time for what we would have predicted
trainpreds = np.insert(trainpreds[:-1], 0, 1) * targstd + targmean
targets = pd.DataFrame(targets)
targets['econFactor'] = trainpreds

traindata = traindata.join(targets.econFactor,on='saledate').fillna(1)

#==============================================================================
# PREDICTION
#==============================================================================
inputs = testdata.drop(['saledate'],axis=1)
testdata['SalePrice'] = pricemodel.predict(inputs)

# predict economic factor based on external variables
SalePrice = pd.DataFrame(testdata.groupby(['saledate']).SalePrice.mean())
totData = extData.diff().join(SalePrice,how='inner')
totData = pd.ewma(totData, span=12)

# use SVM regression for economic model
# tapped delayed lines so that we are actually forecasting, not cheating
inputsd = totData[['econ1','econ2','econ3']]
inputs = pd.concat([inputsd.shift(1), inputsd.shift(2), inputsd.shift(3), inputsd.shift(4), \
                inputsd.shift(7), inputsd.shift(13), inputsd.shift(25)], axis=1).fillna(0)
econFactors = econ.predict(inputs)

testdata.econFactor = 1

# apply factors on coarse-scale group price and correct for fine scale
testdata.SalePrice = np.exp(testdata.econFactor * testdata.SalePrice)

write_submission("submission.csv", testdata.SalePrice, data_path, submission_path)
