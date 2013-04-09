import os
import numpy as np
from pylearn2.utils import serial

# load configuration file and train model
train_obj = serial.load_train_file('/home/nico/Code/kaggle/GenderWrite/gw_maxout.yaml'
train_obj.main_loop()

# generate model output

# reshape model output so that shape[0] = no. of pages

# save
