import pandas as pd
import numpy as np

import utils
import data
import models

# Read file
csv_file_path = '~/Downloads/enb0.csv'
df = data.readCSV(csv_file_path, ';')

# Trim parameters and set dataframe
df = df[['dl_brate', 'ul_brate', 'system_load']]
X = df[['ul_brate', 'dl_brate']]

# Raw Data
# Get splits and estimator
X_train, X_test = data.getTrainTestSplit(X, .2)
isoForest = models.getIsoForest(0.05)

# Fit estimator
models.fitIsoForest(isoForest, X_train)

# Get predict data
X_test_results = X_test.copy()
X_test_results['anomaly'] = models.predictIsoForest(isoForest, X_test)


# Get train data
X_train_results = X_train.copy()
X_train_results['anomaly'] = models.predictIsoForest(isoForest, X_train)



# Derivative Data
# Get data and set dataframe
deriv_df = data.getDiff(df)
D = deriv_df[['ul_brate', 'dl_brate']]

# Get splits and estimator
D_train, D_test = data.getTrainTestSplit(D, .2)

# Fit estimator
models.fitIsoForest(isoForest, D_train)

# Get predict data
D_test_results = D_test.copy()
D_test_results['anomaly'] = models.predictIsoForest(isoForest, D_test)


# Get train data
D_train_results = D_train.copy()
D_train_results['anomaly'] = models.predictIsoForest(isoForest, D_train)

# utils.plotIsoResultsFull(X_test_results, X_train_results, D_test_results, D_train_results, X.columns[0], X.columns[1])

# 3 feature dimension reduction
df_sys = df.copy()

data.addRatioFeature(df_sys, 'ul_brate', 'system_load')
data.addRatioFeature(df_sys, 'dl_brate', 'system_load')

S = df_sys[['ul_brate / system_load', 'dl_brate / system_load']]

# Raw Data
# Get splits and estimator
S_train, S_test = data.getTrainTestSplit(S, .2)

# Fit estimator
models.fitIsoForest(isoForest, S_train)

# Get predict data
S_test_results = S_test.copy()
S_test_results['anomaly'] = models.predictIsoForest(isoForest, S_test)

# Get train data
S_train_results = S_train.copy()
S_train_results['anomaly'] = models.predictIsoForest(isoForest, S_train)

# Derivative Data
# Get data and set dataframe
deriv_df_sys = data.getDiff(df_sys)
SD = deriv_df_sys[['ul_brate / system_load', 'dl_brate / system_load']]

# Get splits and estimator
SD_train, SD_test = data.getTrainTestSplit(SD, .2)

# Fit estimator
models.fitIsoForest(isoForest, SD_train)

# Get predict data
SD_test_results = SD_test.copy()
SD_test_results['anomaly'] = models.predictIsoForest(isoForest, SD_test)


# Get train data
SD_train_results = SD_train.copy()
SD_train_results['anomaly'] = models.predictIsoForest(isoForest, SD_train)

utils.plotIsoResultsFull(S_test_results, S_train_results, SD_test_results, SD_train_results, S.columns[0], S.columns[1], fontsize=8)