import matplotlib.pyplot as plt
import data
import models

def plotIsoForest(X, xParam, yParam, state, raw=True):
    plt.scatter(X.loc[X['anomaly'] == 1, xParam], X.loc[X['anomaly'] == 1, yParam],
                color='blue', label='Normal')
    plt.scatter(X.loc[X['anomaly'] == -1, xParam], X.loc[X['anomaly'] == -1, yParam],
                color='red', label='Anomaly')

    plt.title('%s vs %s %s (%s)' %(xParam, yParam, state, 'raw' if raw else 'derivative'))
    plt.xlabel(xParam)
    plt.ylabel(yParam)
    plt.legend()
    plt.show()

def plotIsoResults(X, D, xParam, yParam, state):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,4))

    # Raw Data Graphs
    ax1.scatter(X.loc[X['anomaly'] == 1, xParam], X.loc[X['anomaly'] == 1, yParam],
               color='blue', label='Normal')
    ax1.scatter(X.loc[X['anomaly'] == -1, xParam], X.loc[X['anomaly'] == -1, yParam],
               color='red', label='Anomaly')

    ax1.set_title('%s vs %s %s (%s)' %(xParam, yParam, state, 'raw'))
    ax1.set_xlabel(xParam)
    ax1.set_ylabel(yParam)
    ax1.legend()


    # Deriv Data Graphs
    ax2.scatter(D.loc[D['anomaly'] == 1, xParam], D.loc[D['anomaly'] == 1, yParam],
               color='blue', label='Normal')
    ax2.scatter(D.loc[D['anomaly'] == -1, xParam], D.loc[D['anomaly'] == -1, yParam],
               color='red', label='Anomaly')

    ax2.set_title('%s vs %s %s (%s)' % (xParam, yParam, state, 'derivative'))
    ax2.set_xlabel(xParam)
    ax2.set_ylabel(yParam)
    ax2.legend()

    plt.show()

def plotIsoResultsFull(X_test, X_train, D_test, D_train, xParam, yParam, **kwargs):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10,8), sharex='col', sharey='col')
    fontSize = kwargs.pop('fontsize', 12)

    # Raw Data Graphs Predict
    ax1.scatter(X_test.loc[X_test['anomaly'] == 1, xParam], X_test.loc[X_test['anomaly'] == 1, yParam],
               color='blue', label='Normal')
    ax1.scatter(X_test.loc[X_test['anomaly'] == -1, xParam], X_test.loc[X_test['anomaly'] == -1, yParam],
               color='red', label='Anomaly')

    ax1.set_title('%s vs %s %s (%s)' %(xParam, yParam, 'predict', 'raw'), fontsize=fontSize)
    ax1.set_xlabel(xParam)
    ax1.set_ylabel(yParam)
    ax1.legend()


    # Deriv Data Graphs Predict
    ax2.scatter(D_test.loc[D_test['anomaly'] == 1, xParam], D_test.loc[D_test['anomaly'] == 1, yParam],
               color='blue', label='Normal')
    ax2.scatter(D_test.loc[D_test['anomaly'] == -1, xParam], D_test.loc[D_test['anomaly'] == -1, yParam],
               color='red', label='Anomaly')

    ax2.set_title('%s vs %s %s (%s)' % (xParam, yParam, 'predict', 'derivative'), fontsize=fontSize)
    ax2.set_xlabel(xParam)
    ax2.set_ylabel(yParam)
    ax2.legend()

    # Raw Data Graphs Train
    ax3.scatter(X_train.loc[X_train['anomaly'] == 1, xParam], X_train.loc[X_train['anomaly'] == 1, yParam],
                color='blue', label='Normal')
    ax3.scatter(X_train.loc[X_train['anomaly'] == -1, xParam], X_train.loc[X_train['anomaly'] == -1, yParam],
                color='red', label='Anomaly')

    ax3.set_title('%s vs %s %s (%s)' % (xParam, yParam, 'train', 'raw'), fontsize=fontSize)
    ax3.set_xlabel(xParam)
    ax3.set_ylabel(yParam)
    ax3.legend()

    # Deriv Data Graphs Train
    ax4.scatter(D_train.loc[D_train['anomaly'] == 1, xParam], D_train.loc[D_train['anomaly'] == 1, yParam],
                color='blue', label='Normal')
    ax4.scatter(D_train.loc[D_train['anomaly'] == -1, xParam], D_train.loc[D_train['anomaly'] == -1, yParam],
                color='red', label='Anomaly')

    ax4.set_title('%s vs %s %s (%s)' % (xParam, yParam, 'train', 'derivative'), fontsize=fontSize)
    ax4.set_xlabel(xParam)
    ax4.set_ylabel(yParam)
    ax4.legend()

    plt.subplots_adjust(**kwargs)
    plt.show()

def graphPlots(enbNum, csvPath, csvName):
    for i in range(0, enbNum, 1):
        csv_file_path = '%s%s%i.csv' %(csvPath, csvName, i)
        df = data.readCSV(csv_file_path, ';')

        # Trim parameters and set dataframe
        df = df[['dl_brate', 'ul_brate', 'system_load']]
        X = df[['ul_brate', 'dl_brate']]

        # Raw Data
        # Get splits and estimator
        isoForest = models.getIsoForest(0.05)
        X_train, X_test = data.getTrainTestSplit(X, .2)
        X_train_results, X_test_results = models.returnAnomalyDF(X_train, X_test)

        # Derivative Data
        # Get data and set dataframe
        deriv_df = data.getDiff(df)
        D = deriv_df[['ul_brate', 'dl_brate']]

        # Get splits and estimator
        D_train, D_test = data.getTrainTestSplit(D, .2)
        D_train_results, D_test_results = models.returnAnomalyDF(D_train, D_test)

        plotIsoResultsFull(X_test_results, X_train_results, D_test_results, D_train_results, X.columns[0],
                                 X.columns[1])

        # 3 feature dimension reduction
        df_sys = df.copy()

        data.addRatioFeature(df_sys, 'ul_brate', 'system_load')
        data.addRatioFeature(df_sys, 'dl_brate', 'system_load')

        S = df_sys[['ul_brate / system_load', 'dl_brate / system_load']]

        # Raw Data
        # Get splits and estimator
        S_train, S_test = data.getTrainTestSplit(S, .2)
        S_train_results, S_test_results = models.returnAnomalyDF(S_train, S_test)

        # Derivative Data
        # Get data and set dataframe
        deriv_df_sys = data.getDiff(df_sys)
        SD = deriv_df_sys[['ul_brate / system_load', 'dl_brate / system_load']]
        SD_train, SD_test = data.getTrainTestSplit(SD, .2)
        SD_train_results, SD_test_results = models.returnAnomalyDF(SD_train, SD_test)

        plotIsoResultsFull(S_test_results, S_train_results, SD_test_results, SD_train_results, S.columns[0],
                                 S.columns[1], fontsize=8)

