from dateutil.parser import parse
import pandas as pd
import numpy as np
import os
from sklearn import ensemble, cross_validation, metrics

DATA_PATH = '/home/nico/datasets/Kaggle/Bulldozers/'


def get_train_test_df():
    return pd.read_csv(os.path.join(DATA_PATH, "Train.csv"), converters={"saledate": parse}), \
            pd.read_csv(os.path.join(DATA_PATH, "Valid.csv"), converters={"saledate": parse})

def write_submission(submission_name, predictions):
    _, test = get_train_test_df()
    test = test.join(pd.DataFrame({"SalePrice": predictions}))
    test[["SalesID", "SalePrice"]].to_csv(os.path.join(DATA_PATH,
        submission_name), index=False)

def get_date_dataframe(date_column):
    return pd.DataFrame({
        "SaleYear": [d.year for d in date_column],
        "SaleMonth": [d.month for d in date_column],
        "SaleDay": [d.day for d in date_column]
        }, index=date_column.index)

train, test = get_train_test_df()

columns = set(train.columns)
columns.remove("SalesID")
columns.remove("SalePrice")
columns.remove("saledate")

train_fea = get_date_dataframe(train["saledate"])
test_fea = get_date_dataframe(test["saledate"])

for col in columns:
    dropcols = ['MachineID','Turbocharged','Grouser_Tracks','Pad_Type','Backhoe_Mounting','Grouser_Type', \
                'Blade_Type','Stick','auctioneerID','datasource','Forks','Pattern_Changer', \
                'Differential_Type','Thumb','Hydraulics_Flow','Steering_Controls', \
                'Track_Type','Undercarriage_Pad_Width','Coupler_System','Travel_Controls', \
                'Ripper','Coupler','Hydraulics','fiModelSeries','ProductGroupDesc', \
                'Tip_Control','Scarifier','Blade_Extension','Drive_System','fiModelDesc','Stick_Length', \
                'PrimaryLower','fiProductClassDesc','fiManufacturerDesc','Enclosure_Type','Pushblock',\
                'SaleDay','state','SaleMonth','fiModelDescriptor']
    if col not in dropcols:
        if train[col].dtype == np.dtype('object'):
            s = np.unique(train[col].fillna(-1).values)
            mapping = pd.Series([x[0] for x in enumerate(s)], index = s)
            train_fea = train_fea.join(train[col].map(mapping).fillna(-1))
            test_fea = test_fea.join(test[col].map(mapping).fillna(-1))
        else:
            train_fea = train_fea.join(train[col].fillna(0))
            test_fea = test_fea.join(test[col].fillna(0))

train_fea['age'] = train_fea.SaleYear - train_fea.YearMade
test_fea['age'] = test_fea.SaleYear - test_fea.YearMade
train_fea.age[train_fea.age.isnull()] = 0
test_fea.age[test_fea.age.isnull()] = 0
train_fea = train_fea.drop(['YearMade'], axis=1)
test_fea = test_fea.drop(['YearMade'], axis=1)

rf = ensemble.RandomForestRegressor(n_estimators=250, n_jobs=-1, compute_importances = True)

# get cross-validation score
cv = cross_validation.KFold(len(train["SalePrice"]), n_folds=5)
scores = cross_validation.cross_val_score(rf, train_fea, train["SalePrice"], \
            cv=cv, n_jobs=1, score_func=metrics.r2_score)
print "Cross-validation accuracy on the training set:"
print "%0.3f (+/-%0.03f)" % (scores.mean(), scores.std() / 2)

# now fit
rf.fit(train_fea, train["SalePrice"])
predictions = rf.predict(test_fea)
imp = sorted(zip(train_fea.columns, rf.feature_importances_), key=lambda tup: tup[1], reverse=True)
for fea in imp:
    print(fea)

write_submission("submission.csv", predictions)
