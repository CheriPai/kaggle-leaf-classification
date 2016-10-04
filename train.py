import pandas as pd
from sklearn import cross_validation, preprocessing
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score, log_loss


leaves = pd.read_csv("data/train.csv")
leaves = leaves.drop(["id"], axis=1)

labels = pd.read_csv("data/sample_submission.csv")
labels = labels.drop(["id"], axis=1).columns.values
le = preprocessing.LabelEncoder()
le.fit(labels)

X = leaves.drop(["species"], axis=1).values
y = leaves.species
y = le.transform(y)

sss = StratifiedShuffleSplit(y, 1, test_size=0.2)
for train_index, test_index in sss:
    X_train, X_val = X[train_index], X[test_index]
    y_train, y_val = y[train_index], y[test_index]

clf = RandomForestClassifier(n_estimators=200, max_features="log2", n_jobs=-1)
clf.fit(X_train, y_train)

predictions = clf.predict(X_val)
print("Accuracy: %.4f" % accuracy_score(y_val, predictions))
predictions = clf.predict_proba(X_val)
print("Log loss: %.4f" % log_loss(y_val, predictions))
print("Saving classifier to data/classifier.pkl")
joblib.dump(clf, "data/classifier.pkl")
