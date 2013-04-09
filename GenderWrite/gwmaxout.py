import numpy as np
from pylearn2.utils import serial

DATA_DIR = '/home/nico/Code/kaggle/GenderWrite/'

# load configuration file and train model
train_obj = serial.load_train_file(DATA_DIR+'gwmaxout.yaml'
train_obj.main_loop()

# generate model output
DATA_DIR+'gw_preprocessed_test.pkl'

# reshape model output so that shape[0] = no. of pages

# save
np.save(DATA_DIR+'maxout_train.npy', trainout)
np.save(DATA_DIR+'maxout_test.npy', testout)
