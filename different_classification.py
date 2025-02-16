import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

heart_disease = pd.read_csv("Data/heart-disease.csv")
heart_disease.head()

X = heart_disease.drop(["target"],axis=1)

y = heart_disease["target"]
print(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y)

print(X_train.shape,y_train.shape,X_test.shape,y_test.shape)

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()

clf.fit(X_train, y_train)

y_preds = clf.predict(X_test)

clf.score(X_train,y_train)

clf.score(X_test,y_test)

from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

models = {"LinearSVC": LinearSVC(),
          "KNN": KNeighborsClassifier(),
          "SVC": SVC(),
          "LogisticRegression": LogisticRegression(),
          "RandomForestClassifier": RandomForestClassifier()}

results = {}
for model_name, model in models.items():
    model.fit(X_train, y_train)
    results[model_name] = model.score(X_test, y_test)

print(results)

np.random.seed(42)

for model_name, model in models.items():
    model.fit(X_train, y_train)
    results[model_name] = model.score(X_test, y_test)

print(results)

results_df = pd.DataFrame(results.values(),
                          results.keys(),
                          columns=["accuracy"])

results_df.plot.bar()

#hyper tuning
log_reg_grid = {"C": np.logspace(-4, 4, 20),
                "solver": ["liblinear"]}
np.random.seed(42)
from sklearn.model_selection import RandomizedSearchCV
rs_log_reg = RandomizedSearchCV(estimator=LogisticRegression(),
                                param_distributions=log_reg_grid,
                                cv=5,
                                n_iter=5,
                                verbose=True)
rs_log_reg.fit(X_train, y_train);
print(rs_log_reg.best_params_)
rs_log_reg.score(X_test, y_test)

clf = LogisticRegression(**rs_log_reg.best_params_)
clf.fit(X_train, y_train);

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import RocCurveDisplay

y_pred=clf.predict(X_test)
print(y_pred)

confusion_matrix(y_test,y_pred)

import seaborn as sns


def plot_conf_mat(y_test, y_preds):
    """
    Plots a confusion matrix using Seaborn's heatmap().
    """
    fig, ax = plt.subplots(figsize=(3, 3))
    ax = sns.heatmap(confusion_matrix(y_test, y_preds),
                     annot=True,  # Annotate the boxes
                     cbar=False)
    plt.xlabel("True label")
    plt.ylabel("Predicted label")

    # Fix the broken annotations (this happened in Matplotlib 3.1.1)
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5);


plot_conf_mat(y_test, y_preds)
print(classification_report(y_test, y_preds))
precision_score(y_test, y_preds)
recall_score(y_test, y_preds)
f1_score(y_test, y_preds)