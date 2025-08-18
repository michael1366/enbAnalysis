from sklearn.model_selection import train_test_split
import pandas as pd

def getTrainTestSplit(X, testSize):
    return train_test_split(X, test_size=testSize, random_state=42)

def readCSV(filePath, delim=','):
    return pd.read_csv(filePath, delimiter=delim)

def getDiff(df):
    ddf = df.diff()
    return ddf.dropna()

def addRatioFeature(df, numerator, denominator):
    def rule(row):
        return row[numerator] / row[denominator] if row[denominator] != 0 else 0

    df['%s / %s' %(numerator, denominator)] = df.apply(rule, axis=1)