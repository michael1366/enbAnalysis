import matplotlib.pyplot as plt

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