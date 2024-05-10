import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import GridSearchCV

data = pd.read_csv('/content/users_behavior.csv', index_col=0)
data.isna().sum() #пропусков нет
X = data.drop('is_ultra', axis=1)
y = data['is_ultra']
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

log_reg = LogisticRegression()
knn = KNeighborsClassifier()
rf = RandomForestClassifier()

param_grid_log_reg = {'solver': ['liblinear', 'lbfgs'],
                      'max_iter': [100, 200, 400]}
param_grid_knn = {'metric': ['cosine', 'euclidean'],
                  'n_neighbors': [5, 7, 9]}
param_grid_rf = {'n_estimators': [10, 15, 20],
                 'max_depth': [10, 20, 30],
                 'criterion': ['gini', 'entropy'],
                 'min_samples_split': [2, 5, 10],
                 'min_samples_leaf': [1, 2, 5]}

grid_search_log_reg = GridSearchCV(estimator=log_reg, param_grid=param_grid_log_reg, cv=5)
grid_search_knn = GridSearchCV(estimator=knn, param_grid=param_grid_knn, cv=5)
grid_search_rf = GridSearchCV(estimator=rf, param_grid=param_grid_rf, cv=5)
grid_search_log_reg.fit(X_train, y_train)
grid_search_knn.fit(X_train, y_train)
grid_search_rf.fit(X_train, y_train)
best_log_reg = grid_search_log_reg.best_params_
best_knn = grid_search_knn.best_params_
best_rf = grid_search_rf.best_params_

b_solver, b_max_iter = best_log_reg['solver'], best_log_reg['max_iter']
b_log_reg = LogisticRegression(solver=b_solver, max_iter=b_max_iter)
b_log_reg.fit(X_train, y_train)
y_pred_log_reg = b_log_reg.predict(X_test)
b_metric, b_n_neighbors = best_knn['metric'], best_knn['n_neighbors']
b_knn = KNeighborsClassifier(metric=b_metric, n_neighbors=b_n_neighbors)
b_knn.fit(X_train, y_train)
y_pred_knn = b_knn.predict(X_test)
b_n_estimators, b_max_depth, b_criterion, b_min_samples_split, b_min_samples_leaf = best_rf['n_estimators'], best_rf['max_depth'], best_rf['criterion'], best_rf['min_samples_split'], best_rf['min_samples_leaf']
b_rf = RandomForestClassifier(n_estimators=b_n_estimators, max_depth=b_max_depth, criterion=b_criterion, min_samples_split=b_min_samples_split, min_samples_leaf=b_min_samples_leaf)
b_rf.fit(X_train, y_train)
y_pred_rf = b_rf.predict(X_test)

ac_log_reg = accuracy_score(y_test, y_pred_log_reg)
f1_log_reg = f1_score(y_test, y_pred_log_reg)
ac_knn = accuracy_score(y_test, y_pred_knn)
f1_knn = f1_score(y_test, y_pred_knn)
ac_rf = accuracy_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf)

#Таблица с валидацией рассмотренных моделей

results = [['accuracy', ac_log_reg, ac_knn, ac_rf],
           ['f1_score', f1_log_reg, f1_knn, f1_rf]]
result = pd.DataFrame(results, columns=['', 'logistic_regression', 'k_nearest_neighbors', 'random_forest'])
result

#Для дополнительной оценки лучшей модели

!pip install scikit-plot
from scikitplot.metrics import plot_confusion_matrix

plot_confusion_matrix(y_test, y_pred_rf, normalize=True)