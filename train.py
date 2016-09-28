import pandas as pd
from sklearn import cross_validation, preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib


leaves = pd.read_csv("data/train.csv")
leaves = leaves.drop(["id"], axis=1)

labels = pd.read_csv("data/sample_submission.csv")
labels = labels.drop(["id"], axis=1).columns.values
le = preprocessing.LabelEncoder()
le.fit(labels)

X = leaves.drop(["species"], axis=1).values
y = leaves.species.values
y = le.transform(y)
X_train, X_val, y_train, y_val = cross_validation.train_test_split(
    X, y, test_size=0.2, random_state=0)

clf = RandomForestClassifier(n_estimators=100, max_features="log2", n_jobs=-1)
clf.fit(X_train, y_train)

print("Accuracy: %.4f" % clf.score(X_val, y_val))
print("Saving to data/classifier.pkl")
joblib.dump(clf, "data/classifier.pkl")
