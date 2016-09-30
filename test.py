import numpy as np
import pandas as pd
from sklearn.externals import joblib


leaves = pd.read_csv("data/test.csv")
X = leaves.drop(["id"], axis=1).values

labels = pd.read_csv("data/sample_submission.csv").columns.values

clf = joblib.load("data/classifier.pkl")
predictions = clf.predict_proba(X)
predictions = np.concatenate((np.array([leaves.id]).T, predictions), axis=1)
