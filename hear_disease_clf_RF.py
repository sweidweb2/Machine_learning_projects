import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

heart_disease = pd.read_csv("Data/heart-disease.csv")
heart_disease.head()

X = heart_disease.drop(["target"],axis=1)

y = heart_disease["target"]
X

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y)

X_train.shape,y_train.shape,X_test.shape,y_test.shape,

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()

clf.fit(X_train, y_train)

y_preds = clf.predict(X_test)

clf.score(X_train,y_train)

clf.score(X_test,y_test)