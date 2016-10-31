# Useful starting lines
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
%load_ext autoreload
%autoreload 2
from helpers import *
from plots import *
import datetime
import pandas as pd
from proj1_helpers import *

from proj1_helpers import *
DATA_TRAIN_PATH = 'train.csv' # TODO: download train data and supply path here 
y, tX, ids = load_csv_data(DATA_TRAIN_PATH, sub_sample=True)

#do stuff

DATA_TEST_PATH = 'test.csv' # TODO: download train data and supply path here 
_, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)
mean_tX_test, mean_mean, mean__std = standardize(undefToMeanMean(tX_test), mean_x=None, std_x=None)
OUTPUT_PATH = 'mean_log_pen_10-4.csv' # TODO: fill in desired name of output file for submission
y_pred = predict_labels(reg_w, mean_tX_test)
create_csv_submission(ids_test, y_pred, OUTPUT_PATH)