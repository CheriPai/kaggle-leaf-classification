import pandas as pd
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier


leaves = pd.read_csv("data/train.csv")
leaves = leaves.drop(["id"], axis=1)

X = leaves.drop(["species"], axis=1).values
y = leaves.species.values
X_train, X_val, y_train, y_val = cross_validation.train_test_split(
    X, y, test_size=0.2, random_state=0)

clf = RandomForestClassifier()
clf.fit(X_train, y_train)

print("Accuracy: %.4f" % clf.score(X_val, y_val))
