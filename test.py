import pandas as pd
from sklearn.externals import joblib


leaves = pd.read_csv("data/test.csv")
X = leaves.drop(["id"], axis=1).values

clf = joblib.load("data/classifier.pkl")
predictions = clf.predict_proba(X)
