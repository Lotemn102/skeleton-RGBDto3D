import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from angles_to_age.data_reader import read
from angles_to_age.preprocess import normalize, to_binary

def correlation_matrix(train, test):
    x_train = pd.DataFrame(train)
    x_test = pd.DataFrame(test)

    ticks = ['alpha_1', 'alpha_2', 'alpha_3', 'alpha_4', 'label']

    corr = x_train.corr()
    sns.heatmap(corr, xticklabels=ticks, yticklabels=ticks)
    plt.show()
    plt.close()

    corr = x_test.corr()
    sns.heatmap(corr, xticklabels=ticks, yticklabels=ticks)
    plt.show()

def plot_data(x_train, y_train):
    # Sub-sample 3 features
    sub_features_indices = [
        [0, 1, 2],
        [1, 2, 3],
        [0, 1, 3],
        [0, 2, 3]
    ]

    x_coor = []
    y_coor = []
    z_coor = []
    labels = []

    for indices in sub_features_indices:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        for i, sample in enumerate(x_train):

            if i % 5 == 0:
                x = sample[indices[0]]
                y = sample[indices[1]]
                z = sample[indices[2]]
                label = 0 if y_train[i] <= 30 else 1 # 0 young, 1 old

                if label == 0:
                    ax.scatter(x, y, z, label='young', color='r')
                else:
                    ax.scatter(x, y, z, label='old', color='g')

                x_coor.append(x)
                y_coor.append(y)
                z_coor.append(z)
                labels.append(label)

        plt.show()
        plt.close()


if __name__ == "__main__":
    x_train, x_test, y_train, y_test = read()

    train = np.append(x_train, np.array([y_train]).T, axis=1)
    test = np.append(x_test, np.array([y_test]).T, axis=1)
    train, test = normalize(train, test)
    correlation_matrix(train, test)

    x_train, x_test = normalize(x_train, x_test)
    plot_data(x_train, y_train)
