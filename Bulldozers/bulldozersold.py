#==============================================================================
# 
# Training:
# 1. original prices --> train pricemodel on traindata --> take residuals (as fractions of modeled price) to extract further meaning
# 2. train econmodel with external data on residuals --> per month, an econFactor  --> take residuals (still as fractions)
# 3. train deprmodel based on exponential regression on age and residuals --> per age, a deprFactor --> ignore residuals
# 
# Testing:
# 1. use pricemodel to predict prices, based on valdata
# 2. use econmodel to predict econFactor
# 3. use age to predict deprFactor
# 4. multiply these 3
#==============================================================================


import pandas as pd
import numpy as np
from sklearn import linear_model, svm, ensemble, cross_validation, metrics
from datetime import datetime
import pylab

econmodel = False
deprmodel = False

# Load the vehicle data
path = '/home/nico/temp/Kaggle data/Bulldozers/'
rawtraindata = pd.read_csv(path + 'Train.csv')
appendix = pd.read_csv(path + 'Machine_Appendix.csv')
rawvaldata = pd.read_csv(path + 'Valid.csv')

def prep_data(d, app, train):
    # drop unused columns from main data set, instead use the appendix
    d = d.drop(['ModelID','datasource','YearMade','fiModelDesc','fiBaseModel', \
    'fiSecondaryDesc','fiModelSeries','fiModelDescriptor','fiProductClassDesc','ProductGroup', \
    'ProductGroupDesc'], axis=1)
    # drop columns that are in fact duplicates from appendix
    app = app.drop(['fiModelDesc','fiProductClassDesc','ProductGroupDesc','fiManufacturerDesc'], axis=1)
    
    d = pd.merge(d, app, on='MachineID', how='left')
    
    def convert_dates(dtstr):
        # European format
        dtstr = dtstr.replace('0:', '00:')
        if dtstr.count('-'):
            dt = datetime.strptime(dtstr, '%d-%m-%Y %H:%M')
        if dtstr.count('/'):
            dt = datetime.strptime(dtstr, '%m/%d/%Y %H:%M')
        # monthly time scale; no cheating, so round to first day of NEXT month
        # so price on 1-1-2012 means the mean price over december 2011
        dyear, dmonth = divmod(dt.month + 1, 12)
        if dmonth == 0:
            dmonth = 12
            dyear = dyear - 1
        dt = datetime(dt.year + dyear, dmonth, 1).date()
        return dt
    
    d.saledate = d.saledate.map(convert_dates)
    
    # fill missing values
    def fillcond_year(group):
        mask = (group < 1940) | (group == 9999) | (group.isnull())
        group[mask] = group[~mask].median()
        return group
    years = d.groupby(['ModelID']).MfgYear
    d.MfgYear = years.transform(fillcond_year)
    years = d.groupby(['ProductGroup']).MfgYear
    d.MfgYear = years.transform(fillcond_year)
    
    d['age'] = pd.DatetimeIndex(d.saledate).year - d.MfgYear
    # fill negative values
    def fillcond_age(group):
        mask = ((group < 0) | group.isnull())
        group[mask] = group[~mask].median()
        return group
    years = d.groupby(['ModelID']).age
    d.age = years.transform(fillcond_age)
    years = d.groupby(['ProductGroup']).age
    d.age = years.transform(fillcond_age)
    d.age[d.age.isnull()] = np.median(d.age)
    d = d.drop(['MfgYear'], axis=1)
    
    def fillcond_upper(group):
        mask = (group.isnull())
        group[mask] = group[~mask].median()
        return group
    uppers = d.groupby(['ModelID']).PrimaryUpper
    d.PrimaryUpper = uppers.transform(fillcond_upper)
    uppers = d.groupby(['ProductGroup']).PrimaryUpper
    d.PrimaryUpper = uppers.transform(fillcond_upper)
    d.PrimaryUpper[d.PrimaryUpper.isnull()] = np.median(d.PrimaryUpper)
    
    def fillcond_lower(group):
        mask = (group.isnull())
        group[mask] = group[~mask].median()
        return group
    lowers = d.groupby(['ModelID']).PrimaryLower
    d.PrimaryLower = lowers.transform(fillcond_lower)
    uppers = d.groupby(['ProductGroup']).PrimaryLower
    d.PrimaryLower = lowers.transform(fillcond_lower)
    d.PrimaryLower[d.PrimaryLower.isnull()] = np.median(d.PrimaryLower)
    
    d.MachineHoursCurrentMeter[d.MachineHoursCurrentMeter.isnull()] = 0
    d.auctioneerID[d.auctioneerID.isnull()] = np.median(d.auctioneerID)
    
    # choose names of features that will be divided into categories
    columns = set(d.columns)
    columns.remove('SalesID')
    
    # initialize with a column so that we can use the index
    features = pd.DataFrame(d.SalesID)
    # over all features that are to be categorized...
    for col in columns:
        # see if column is actually an object, i.e. strings/non-numeric columns
        if d[col].dtype == np.dtype('object') and col != 'saledate':
            # get list of unique values in this column
            s = np.unique(d[col].values)
            # enumerate over the unique values list generates a list with numbers for the categories
            # make a Series with index = text value, value = number
            mapping = pd.Series([x[0] for x in enumerate(s)], index=s)
            # map SeriesB's values onto SeriesA's index using their corresponding index(B) / values(A)
            # then left join column on index into the main dataframe
            features = features.join(d[col].map(mapping))
        # this is for numeric columns
        else:
            features = features.join(d[col])
            
    return features

traindata = prep_data(rawtraindata, appendix, 1)
valdata = prep_data(rawvaldata, appendix, 0)
valdata = traindata[traindata.saledate > datetime(2011,01,01).date()].drop('SalePrice', axis=1)
refdata = traindata[traindata.saledate > datetime(2011,01,01).date()][['SalesID','SalePrice']]

# convert prices to log scale for RMSLE
traindata['logSalePrice'] = np.log(traindata.SalePrice)

# select inputs
inputcols = ['ProductGroup', 'ModelID', 'PrimarySizeBasis', 'PrimaryUpper', 'fiManufacturerID','age']
inputs = traindata[inputcols]

#inputs = traindata
#inputs = inputs.drop(['SalesID','SalePrice','MachineID','saledate','logSalePrice','age'], axis=1)

targets = traindata.logSalePrice

# build price-predicting model
#model = ensemble.GradientBoostingRegressor(learning_rate=0.1, n_estimators=100, \
#        subsample=0.9, max_features=2)
pricemodel = ensemble.RandomForestRegressor(n_estimators=50, max_depth=None, \
             max_features="auto", min_samples_split=20)

cv = cross_validation.ShuffleSplit(len(targets), n_iter=10)

scores = cross_validation.cross_val_score(pricemodel, inputs, targets, \
        cv=cv, n_jobs=-1, score_func=metrics.r2_score)
print "Cross-validation accuracy on the training set:"
print "%0.3f (+/-%0.03f)" % (scores.mean(), scores.std() / 2)

pricemodel = pricemodel.fit(inputs, targets)
traindata['estPrice'] = pricemodel.predict(inputs)
traindata['estFactor'] = traindata.logSalePrice / traindata.estPrice

if econmodel:
    # build economy-corrected price model relating relPrice to external data
    # take the most coarse-scale group, others are too sparse for this
    
    # make 2 matching time series
    estFactors = pd.DataFrame(traindata.groupby(['saledate']).estFactor.mean())
    
    ext1Data = pd.read_csv(path+'TTLCONS.csv', names=['saledate','econ1'], skiprows=1)
    ext1Data.saledate = ext1Data.saledate.map(lambda x: datetime.strptime(x, '%Y-%m-%d').date())
    ext2Data = pd.read_csv(path+'HTRUCKSSA.csv', names=['saledate','econ2'], skiprows=1)
    ext2Data.saledate = ext2Data.saledate.map(lambda x: datetime.strptime(x, '%Y-%m-%d').date())
    ext3Data = pd.read_csv(path+'CUSR0000SETA02.csv', names=['saledate','econ3'], skiprows=1)
    ext3Data.saledate = ext3Data.saledate.map(lambda x: datetime.strptime(x, '%Y-%m-%d').date())
    extData = pd.merge(ext1Data,ext2Data,ext3Data,on='saledate')
    
    # shift with 1 time step to align properly with estFactors
    extData = extData.set_index('saledate').shift(1)
    # smooth the stuff out a bit
    extData = pd.rolling_mean(extData, 6)
    
    # training data, exponentially smoothed
    totData = extData.diff().join(estFactors,how='inner')
    totData = pd.ewma(totData, span=12)
    
    # use SVM regression for economic model
    # tapped delayed lines so that we are actually forecasting, not cheating
    inputsd = totData[['econ1','econ2']]
    inputs = pd.concat([inputsd.shift(1), inputsd.shift(2), inputsd.shift(3), inputsd.shift(4), \
                    inputsd.shift(7), inputsd.shift(13), inputsd.shift(25)], axis=1).fillna(0)
    targets = totData.estFactor
    
    # normalize
    inputs = (inputs - inputs.mean()) / inputs.std()
    
    targmean = targets.mean()
    targstd = targets.std()
    targets = (targets - targmean) / targstd
    
    econ = svm.SVR(kernel='rbf', degree=2, gamma=0.0, C=0.1, epsilon=0.2, shrinking=True)
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
    
else:
    traindata['econFactor'] = traindata['estFactor']


# depreciation model, based on exponential model
def depr_model(age, int, base):
    return int * base ** age

# price is already log-transformed, but again for exponential regr.
logFactor = np.log(traindata.econFactor)

# TODO: corrigeren voor eerste bekende prijs (=conditie), zal iteratief na de vorige stap moeten

# now we can use linear regr. and translate the coefficients back to exponential interpretation
deprInput = traindata[['age']]
depr = linear_model.Ridge(alpha=1.0, fit_intercept=True, solver='lsqr')
depr = depr.fit(deprInput, logFactor)

print "Depreciation model fit: %f" % depr.score(deprInput, logFactor)

print "Coefficients: %f, %f" % (np.exp(depr.intercept_), np.exp(depr.coef_[0]))

deprInt = np.exp(depr.intercept_)
deprBase = np.exp(depr.coef_[0])

# plot depr model
pylab.scatter(deprInput[0::100], np.exp(logFactor[0::100]), color='black')
pylab.plot(deprInput[0::100], np.exp(depr.predict(deprInput[0::100])), color='blue')
pylab.show()


# make prediction
#inputs = valdata.drop(['SalesID','MachineID','saledate','age'], axis=1)
inputs = valdata[inputcols]
valdata['estPrice'] = pricemodel.predict(inputs)

if econmodel:
    # predict economic factor based on external variables
    estPrice = pd.DataFrame(valdata.groupby(['saledate']).estPrice.mean())
    totData = extData.diff().join(estPrice,how='inner')
    totData = pd.ewma(totData, span=12)
    
    # use SVM regression for economic model
    # tapped delayed lines so that we are actually forecasting, not cheating
    inputsd = totData[['econ1','econ2']]
    inputs = pd.concat([inputsd.shift(1), inputsd.shift(2), inputsd.shift(3), inputsd.shift(4), \
                    inputsd.shift(7), inputsd.shift(13), inputsd.shift(25)], axis=1).fillna(0)
else:
    valdata['econFactor'] = 1

if deprmodel:
    # calculate depreciation based on age
    # intercept to 1, because no prior knowledge of condition vehicles
    valdata['deprFactor'] = valdata.age.apply(depr_model, args=(deprInt, deprBase))
else:
    valdata['deprFactor'] = 1

# apply factors on coarse-scale group price and correct for fine scale
valdata['SalePrice'] = np.exp(valdata.econFactor * valdata.deprFactor * valdata.estPrice)

# generate submission file
output = valdata[['SalesID', 'SalePrice']].sort(columns='SalesID')

# save to CSV
output.to_csv('/home/nico/Dropbox/Research Projects/Kaggle/Bulldozers/submission.csv', \
                    sep=',', header=True, index=False)
refdata.to_csv('/home/nico/Dropbox/Research Projects/Kaggle/Bulldozers/tsubmission.csv', \
                    sep=',', header=True, index=False)
