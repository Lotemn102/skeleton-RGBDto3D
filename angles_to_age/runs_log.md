## Small dataset
### RandomForest
Best parameters set found on validation set:
{'max_depth': 50, 'n_estimators': 50}

Validation accuracy:
0.823  for {'max_depth': 10, 'n_estimators': 50}
0.831  for {'max_depth': 10, 'n_estimators': 100}
0.836  for {'max_depth': 10, 'n_estimators': 200}
0.837  for {'max_depth': 50, 'n_estimators': 50}
0.835  for {'max_depth': 50, 'n_estimators': 100}
0.828  for {'max_depth': 50, 'n_estimators': 200}
0.828  for {'max_depth': 100, 'n_estimators': 50}
0.830  for {'max_depth': 100, 'n_estimators': 100}
0.835  for {'max_depth': 100, 'n_estimators': 200}
0.827  for {'max_depth': 1000, 'n_estimators': 50}
0.831  for {'max_depth': 1000, 'n_estimators': 100}
0.829  for {'max_depth': 1000, 'n_estimators': 200}

Test accuracy is: 0.6842672413793104

<class 'sklearn.svm._classes.SVC'>
Best parameters set found on validation set:
{'C': 1000, 'kernel': 'linear'}

Validation accuracy:
0.766  for {'C': 1, 'kernel': 'linear'}
0.884  for {'C': 1, 'kernel': 'poly'}
0.832  for {'C': 1, 'kernel': 'rbf'}
0.796  for {'C': 1, 'kernel': 'sigmoid'}
0.819  for {'C': 10, 'kernel': 'linear'}
0.854  for {'C': 10, 'kernel': 'poly'}
0.829  for {'C': 10, 'kernel': 'rbf'}
0.794  for {'C': 10, 'kernel': 'sigmoid'}
0.887  for {'C': 100, 'kernel': 'linear'}
0.841  for {'C': 100, 'kernel': 'poly'}
0.832  for {'C': 100, 'kernel': 'rbf'}
0.720  for {'C': 100, 'kernel': 'sigmoid'}
0.897  for {'C': 1000, 'kernel': 'linear'}
0.829  for {'C': 1000, 'kernel': 'poly'}
0.830  for {'C': 1000, 'kernel': 'rbf'}
0.597  for {'C': 1000, 'kernel': 'sigmoid'}

Test accuracy is: 0.6142241379310345

<class 'sklearn.neighbors._classification.KNeighborsClassifier'>
Best parameters set found on validation set:
{'n_neighbors': 5, 'weights': 'uniform'}

Validation accuracy:
0.821  for {'n_neighbors': 3, 'weights': 'uniform'}
0.823  for {'n_neighbors': 5, 'weights': 'uniform'}
0.820  for {'n_neighbors': 10, 'weights': 'uniform'}
0.781  for {'n_neighbors': 50, 'weights': 'uniform'}
0.755  for {'n_neighbors': 100, 'weights': 'uniform'}

Test accuracy is: 0.7683189655172413

<class 'sklearn.ensemble._weight_boosting.AdaBoostClassifier'>
Best parameters set found on validation set:
{'learning_rate': 1, 'n_estimators': 10}

Validation accuracy:
0.818  for {'learning_rate': 0.5, 'n_estimators': 10}
0.792  for {'learning_rate': 0.5, 'n_estimators': 50}
0.809  for {'learning_rate': 0.5, 'n_estimators': 100}
0.821  for {'learning_rate': 1, 'n_estimators': 10}
0.818  for {'learning_rate': 1, 'n_estimators': 50}
0.810  for {'learning_rate': 1, 'n_estimators': 100}
0.497  for {'learning_rate': 5, 'n_estimators': 10}
0.497  for {'learning_rate': 5, 'n_estimators': 50}
0.497  for {'learning_rate': 5, 'n_estimators': 100}

Test accuracy is: 0.6228448275862069

## Large dataset (taking every second frame)
### RandomForest
Best parameters set found on validation set:
{'max_depth': 50, 'n_estimators': 50}

Validation accuracy:
0.749  for {'max_depth': 10, 'n_estimators': 50}
0.751  for {'max_depth': 10, 'n_estimators': 100}
0.753  for {'max_depth': 10, 'n_estimators': 200}
0.777  for {'max_depth': 50, 'n_estimators': 50}
0.774  for {'max_depth': 50, 'n_estimators': 100}
0.771  for {'max_depth': 50, 'n_estimators': 200}
0.777  for {'max_depth': 100, 'n_estimators': 50}
0.774  for {'max_depth': 100, 'n_estimators': 100}
0.774  for {'max_depth': 100, 'n_estimators': 200}
0.775  for {'max_depth': 1000, 'n_estimators': 50}
0.773  for {'max_depth': 1000, 'n_estimators': 100}
0.771  for {'max_depth': 1000, 'n_estimators': 200}

Test accuracy is: 0.6058394160583942

### SVM
Best parameters set found on validation set:
{'C': 1, 'kernel': 'sigmoid'}

Validation accuracy:
0.621  for {'C': 1, 'kernel': 'linear'}
0.734  for {'C': 1, 'kernel': 'poly'}
0.734  for {'C': 1, 'kernel': 'rbf'}
0.743  for {'C': 1, 'kernel': 'sigmoid'}
0.619  for {'C': 10, 'kernel': 'linear'}
0.717  for {'C': 10, 'kernel': 'poly'}
0.737  for {'C': 10, 'kernel': 'rbf'}
0.743  for {'C': 10, 'kernel': 'sigmoid'}
0.695  for {'C': 100, 'kernel': 'linear'}
0.709  for {'C': 100, 'kernel': 'poly'}
0.706  for {'C': 100, 'kernel': 'rbf'}
0.743  for {'C': 100, 'kernel': 'sigmoid'}
0.735  for {'C': 1000, 'kernel': 'linear'}
0.706  for {'C': 1000, 'kernel': 'poly'}
0.699  for {'C': 1000, 'kernel': 'rbf'}
0.742  for {'C': 1000, 'kernel': 'sigmoid'}

Test accuracy is: 0.5797304884896126

### KNN
Best parameters set found on validation set:
{'n_neighbors': 3, 'weights': 'distance'}

Validation accuracy:
0.762  for {'n_neighbors': 3, 'weights': 'uniform'}
0.763  for {'n_neighbors': 3, 'weights': 'distance'}
0.759  for {'n_neighbors': 5, 'weights': 'uniform'}
0.759  for {'n_neighbors': 5, 'weights': 'distance'}
0.755  for {'n_neighbors': 10, 'weights': 'uniform'}
0.754  for {'n_neighbors': 10, 'weights': 'distance'}
0.727  for {'n_neighbors': 50, 'weights': 'uniform'}
0.733  for {'n_neighbors': 50, 'weights': 'distance'}
0.697  for {'n_neighbors': 100, 'weights': 'uniform'}
0.714  for {'n_neighbors': 100, 'weights': 'distance'}

Test accuracy is: 0.6193149915777653
