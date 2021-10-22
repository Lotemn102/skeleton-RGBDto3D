from sklearn.model_selection import LeaveOneOut, KFold
from sklearn import tree
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.manifold import Isomap
from sklearn.decomposition import KernelPCA
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from angles_to_age.data_reader import read
from angles_to_age.preprocess import normalize, to_binary, shuffle


# RandomForest, KNN, SVM.

if __name__ == "__main__":
    x_train, x_test, y_train, y_test = read()
    x_train, x_test = normalize(x_train, x_test)
    y_train, y_test = to_binary(y_train, y_test)
    x_train, x_test, y_train, y_test = shuffle(x_train, x_test, y_train, y_test)

    X = x_train
    y = np.array([int(e[0]) for e in y_train]) # Each label is tuple of (age, filename). We only need the ages for the training.

    # CV
    loo = LeaveOneOut()
    k_fold = KFold(n_splits=10)

    models = {RandomForestClassifier : {"n_estimators" : [50, 100, 200],
                                        "max_depth" : [10, 50, 100, 1000]},
              SVC : {"C" : [1, 10, 100, 1000],
                     "kernel" : ['linear']},
              KNeighborsClassifier : {'n_neighbors' : [3, 5, 10, 50, 100], 'weights' : ['uniform']}
              }

    models = {KNeighborsClassifier : {'n_neighbors' : [3, 5, 10, 50, 100], 'weights' : ['uniform']}}

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

        false_young = 0
        false_old = 0
        true_young = 0
        true_old = 0
        subjects_mistakes_filenames = []
        y_ = []
        y_hat_ = []

        for i, sample in enumerate(x_test):
            sample = sample.reshape(1, -1)
            y_hat = best_model.predict(sample)
            y_true = int(y_test[i][0])
            filename = y_test[i][1]

            y_.append(y_true)
            y_hat_.append(y_hat)

            if y_hat != y_true:
                subjects_mistakes_filenames.append(filename)
                if y_true == 0:
                    false_young += 1
                else:
                    false_old += 1
            else:
                if y_true == 0:
                    true_young += 1
                else:
                    true_old += 1

            test_scores.append(y_true == y_hat)

        test_acc = 100*round(np.mean(test_scores), 3)
        print("Test accuracy is: {a}%".format(a=test_acc))
        print("In the {n}% wrong classifications, {o}% are old and {y}% are young".format(n=round(100-test_acc, 3),
                                                                                       o=100*round(false_old / (false_old + false_young), 3),
                                                                                       y=100*round(false_young / (false_old + false_young), 2)))
        print("Wrong classifications where in files: {s}".format(s=set(subjects_mistakes_filenames)))

        cm = confusion_matrix(y_true=y_, y_pred=y_hat_)
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] # Normalize the cm
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['young', 'old'])
        disp.plot()
        plt.show()

        print(classification_report(y_true=y_, y_pred=y_hat_))

        print("-------------------------------------------------------")




