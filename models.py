from sklearn.ensemble import IsolationForest

def getIsoForest(contamRatio):
    return IsolationForest(contamination=contamRatio, random_state=42)

def fitIsoForest(isoForest, X):
    isoForest.fit(X)

def predictIsoForest(isoForest, X):
    return isoForest.predict(X)

def returnAnomalyDF(X_train, X_test):
    isoForest = getIsoForest(0.05)
    fitIsoForest(isoForest, X_train)

    X_test_results = X_test.copy()
    X_test_results['anomaly'] = predictIsoForest(isoForest, X_test)

    X_train_results = X_train.copy()
    X_train_results['anomaly'] = predictIsoForest(isoForest, X_train)

    return X_test_results, X_train_results