from sklearn.ensemble import IsolationForest

def getIsoForest(contamRatio):
    return IsolationForest(contamination=contamRatio, random_state=42)

def fitIsoForest(isoForest, X):
    isoForest.fit(X)

def predictIsoForest(isoForest, X):
    return isoForest.predict(X)