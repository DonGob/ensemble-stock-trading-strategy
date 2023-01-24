from lib2to3.pgen2.token import STRING
import pathlib

#import finrl

import pandas as pd
import datetime
import os
#pd.options.display.max_rows = 10
#pd.options.display.max_columns = 10


#PACKAGE_ROOT = pathlib.Path(finrl.__file__).resolve().parent
#PACKAGE_ROOT = pathlib.Path().resolve().parent

#TRAINED_MODEL_DIR = PACKAGE_ROOT / "trained_models"
#DATASET_DIR = PACKAGE_ROOT / "data"

# data
#TRAINING_DATA_FILE = "data/ETF_SPY_2009_2020.csv"
TRAINING_DATA_FILE = "data/dow_30_2009_2020.csv"

now = datetime.datetime.now()
date_time = now.strftime("%m-%d-%Y,%H-%M-%S")
print("TEST NOW:", date_time)
print("TYPE NOW:", type(date_time))
TRAINED_MODEL_DIR = os.path.join("trained_models", date_time)
print("TEST AFTER CREATION:", TRAINED_MODEL_DIR)
os.makedirs(TRAINED_MODEL_DIR)
TURBULENCE_DATA = "data/dow30_turbulence_index.csv"

TESTING_DATA_FILE = "test.csv"


