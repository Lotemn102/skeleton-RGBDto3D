from sklearn.model_selection import LeaveOneOut, KFold
from sklearn import tree
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
import numpy as np

from angles_to_age.data_reader import read
from angles_to_age.preprocess import normalize, to_binary


# RandomForest, KNN, SVM, AdaBoost.

if __name__ == "__main__":
    x_train, x_test, y_train, y_test = read()
    x_train, x_test = normalize(x_train, x_test)
    y_train, y_test = to_binary(y_train, y_test)

    X = x_train
    y = y_train

    # CV
    loo = LeaveOneOut()
    k_fold = KFold(n_splits=10)

    models = {RandomForestClassifier : {"n_estimators" : [50, 100, 200],
                                        "max_depth" : [10, 50, 100, 1000]},
              SVC : {"C" : [1, 10, 100, 1000],
                     "kernel" : ['linear', 'poly', 'rbf', 'sigmoid']},
              KNeighborsClassifier : {'n_neighbors' : [3, 5, 10, 50, 100], 'weights' : ['uniform']},
              AdaBoostClassifier : {'n_estimators' : [10, 50, 100], 'learning_rate' : [0.5, 1, 5]}
              }

    # models = {AdaBoostClassifier : {'n_estimators' : [10, 50, 100], 'learning_rate' : [0.05, 0.5, 1]}}
    scores = []

    for model_name, parameters in models.items():
        print(model_name)
        model = model_name()
        clf = GridSearchCV(estimator=model, param_grid=parameters, cv=k_fold, scoring='accuracy')
        clf.fit(X, y)
        print("Best parameters set found on validation set:")
        print(clf.best_params_)
        print()
        print("Validation accuracy:")
        means = clf.cv_results_['mean_test_score']
        for mean, params in zip(means, clf.cv_results_['params']):
            print("%0.3f  for %r" % (mean, params))
        print()

        # Fit model with best parameters
        best_model = model_name(**clf.best_params_)
        best_model.fit(X, y)

        # Evaluate on the test
        test_scores = []

        for i, sample in enumerate(x_test):
            sample = sample.reshape(1, -1)
            y_hat = best_model.predict(sample)
            y_true = y_test[i]
            test_scores.append(y_true == y_hat)

        print("Test accuracy is: {a}".format(a=np.mean(test_scores)))
        print("-------------------------------------------------------")




