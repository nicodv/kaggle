from dateutil.parser import parse
import pandas as pd
import os

def get_paths():
    """
    Redefine data_path and submissions_path here to run the benchmarks on your machine
    """
    data_path = '/home/nico/temp/Kaggle data/Bulldozers/'
    submission_path = '/home/nico/Dropbox/Research Projects/Kaggle/Bulldozers/'
    return data_path, submission_path

def get_train_df(data_path = None):
    if data_path is None:
        data_path, submission_path = get_paths()
        
    train = pd.read_csv(os.path.join(data_path, "Train.csv"),
        converters={"saledate": parse})
    return train

def get_test_df(data_path = None):
    if data_path is None:
        data_path, submission_path = get_paths()
        
    test = pd.read_csv(os.path.join(data_path, "Valid.csv"),
        converters={"saledate": parse})
    return test

def get_train_test_df(data_path = None):
    return get_train_df(data_path), get_test_df(data_path)

def write_submission(submission_name, predictions, submission_path=None):
    if submission_path is None:
        data_path, submission_path = get_paths()
    
    test = get_test_df()
    test = test.join(pd.DataFrame({"SalePrice": predictions}))
    
    test[["SalesID", "SalePrice"]].to_csv(os.path.join(submission_path,
        submission_name), index=False)
        

import numpy as np
from collections import defaultdict
from sklearn.ensemble import RandomForestRegressor

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
            train_fea = train_fea.join(pd.DataFrame({col: [str_to_categorical[x] for x in train[col]]}, index=train.index))
            test_fea = test_fea.join(pd.DataFrame({col: [str_to_categorical[x] for x in test[col]]}, index=test.index))
        else:
            train_fea = train_fea.join(train[col])
            test_fea = test_fea.join(test[col])

train_fea.MachineHoursCurrentMeter[train_fea.MachineHoursCurrentMeter.isnull()] = 0
train_fea.auctioneerID[train_fea.auctioneerID.isnull()] = np.median(train_fea.auctioneerID)
test_fea.MachineHoursCurrentMeter[test_fea.MachineHoursCurrentMeter.isnull()] = 0
test_fea.auctioneerID[test_fea.auctioneerID.isnull()] = np.median(test_fea.auctioneerID)

rf = RandomForestRegressor(n_estimators=100, n_jobs=-1, compute_importances = True)
rf.fit(train_fea, train["SalePrice"])
predictions = rf.predict(test_fea)
imp = sorted(zip(train_fea.columns, rf.feature_importances_), key=lambda tup: tup[1], reverse=True)
for fea in imp:
    print(fea)

write_submission("random_forest_benchmark.csv", predictions)
