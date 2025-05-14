import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score
import matplotlib.pyplot as plt
from sklearn.svm import SVC, LinearSVC

# https://www.kaggle.com/datasets/ehababoelnaga/anemia-types-classification
df = pd.read_csv("diagnosed_cbc_data_v4.csv")
df.dropna(inplace=True)

# Select columns for X and y
X = df.drop(columns=['LYMp', 'NEUTp', 'LYMn', 'NEUTn', 'Diagnosis'])
y = df['Diagnosis']

# Normalize all X columns
X -= np.average(X)
X /= np.std(X)

# Perform a Grid Search on a Bagging Classifier with a Linear SVC to find the optimal C parameter
estimator = LinearSVC(class_weight='balanced', max_iter=20000)
reg = BaggingClassifier(estimator=estimator)
parameters = {"estimator__C": np.linspace(1, 100, num=5)}
grid_search = GridSearchCV(reg, param_grid = parameters, cv=5, scoring="f1_weighted")
grid_search.fit(X,y)
score_dif = pd.DataFrame(grid_search.cv_results_)
print(score_dif[['param_estimator__C', 'mean_test_score', 'rank_test_score']])

# Fit with the found optimal C value and print the score
c = grid_search.best_params_['estimator__C']
clf = BaggingClassifier(estimator=LinearSVC(C=c, class_weight='balanced', max_iter=20000))
clf.fit(X, y)
y_pred = clf.predict(X)
print(f"Score: {f1_score(y, y_pred, average='weighted'):.3f}")

# Show a confusion matrix with the optimal C
cm = confusion_matrix(y, clf.predict(X), normalize="true")
disp_cm = ConfusionMatrixDisplay(cm, display_labels=clf.classes_)
disp_cm.plot()
plt.show()

# Perform a Grid Search to find the optimal mad depth for a Random Forest
reg = RandomForestClassifier(oob_score=True)
parameters = {"max_depth": [int(x) for x in np.linspace(1, 15, num=5)]}
grid_search = GridSearchCV(reg, param_grid = parameters, cv=5, scoring='f1_weighted')
grid_search.fit(X,y)
score_dif = pd.DataFrame(grid_search.cv_results_)
print(score_dif[['param_max_depth', 'mean_test_score', 'rank_test_score']])

# Fit with the found optimal max depth
max_depth = grid_search.best_params_['max_depth']
clf = RandomForestClassifier(max_depth=max_depth, oob_score=True, verbose=3)
clf.fit(X, y)

# Print the importances of the Random Forest
importances = pd.DataFrame(clf.feature_importances_, index=X.columns)
importances.plot.bar()
plt.show()

# Print the score and OOB score of the optimal Random Forest
print(f"Score: {clf.score(X, y):.3f}")
print(f"OOB Score: {clf.oob_score_:.3f}")

# Print the confusion matrix for the optimal Random Forest
cm = confusion_matrix(y, clf.predict(X), normalize="true")
disp_cm = ConfusionMatrixDisplay(cm, display_labels=clf.classes_)
disp_cm.plot()
plt.show()
