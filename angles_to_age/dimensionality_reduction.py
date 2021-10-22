"""
Several algorithms for dimensionality reduction, so i would be able to visualize the data.
"""
import numpy as np
from sklearn.manifold import Isomap, MDS, TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd

from angles_to_age.data_reader import read
from angles_to_age.preprocess import normalize, to_binary, shuffle


def isomap(x, y):
    model = Isomap(n_components=2)
    proj = model.fit_transform(x)
    proj_df = pd.DataFrame(proj)

    idx_1 = np.where(y == 1)
    idx_0 = np.where(y == 0)

    plt.scatter(proj_df.iloc[idx_1][0], proj_df.iloc[idx_1][1], s=10, c='b', marker="o", label='old')
    plt.scatter(proj_df.iloc[idx_0][0], proj_df.iloc[idx_0][1], s=10, c='r', marker="o", label='young')
    plt.legend()
    plt.show()


def multidimensional_scaling(x, y, metric=True):
    model = MDS(n_components=2, metric=metric)
    proj = model.fit_transform(x)
    proj_df = pd.DataFrame(proj)
    print("here")

    idx_1 = np.where(y == 1)
    idx_0 = np.where(y == 0)

    plt.scatter(proj_df.iloc[idx_1][0], proj_df.iloc[idx_1][1], s=10, c='b', marker="o", label='old')
    plt.scatter(proj_df.iloc[idx_0][0], proj_df.iloc[idx_0][1], s=10, c='r', marker="o", label='young')
    plt.legend()
    plt.show()

def tsne(x, y):
    model = TSNE(n_components=2, learning_rate='auto')
    proj = model.fit_transform(x)
    proj_df = pd.DataFrame(proj)

    idx_1 = np.where(y == 1)
    idx_0 = np.where(y == 0)

    plt.scatter(proj_df.iloc[idx_1][0], proj_df.iloc[idx_1][1], s=10, c='b', marker="o", label='old')
    plt.scatter(proj_df.iloc[idx_0][0], proj_df.iloc[idx_0][1], s=10, c='r', marker="o", label='young')

    plt.legend()
    plt.show()

def pca(x, y):
    model = PCA(n_components=2)
    proj = model.fit_transform(x)
    proj_df = pd.DataFrame(proj)

    idx_1 = np.where(y == 1)
    idx_0 = np.where(y == 0)

    plt.scatter(proj_df.iloc[idx_1][0], proj_df.iloc[idx_1][1], s=10, c='b', marker="o", label='old')
    plt.scatter(proj_df.iloc[idx_0][0], proj_df.iloc[idx_0][1], s=10, c='r', marker="o", label='young')

    plt.legend()
    plt.show()

def plot_all(x, y):
    models = [Isomap(n_components=2), MDS(n_components=2, metric=True), TSNE(n_components=2, learning_rate='auto'),
              PCA(n_components=2)]
    models_names = ['Isomap', 'MDS', 'tSNE', 'PCA']

    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(11, 11))
    axs = axs.ravel()

    for i, model in enumerate(models):
        print(i)
        proj = model.fit_transform(x)
        proj_df = pd.DataFrame(proj)

        idx_1 = np.where(y == 1)
        idx_0 = np.where(y == 0)

        axs[i].scatter(proj_df.iloc[idx_1][0], proj_df.iloc[idx_1][1], s=10, c='b', marker="o", label='old')
        axs[i].scatter(proj_df.iloc[idx_0][0], proj_df.iloc[idx_0][1], s=10, c='r', marker="o", label='young')
        axs[i].legend()
        axs[i].set_title(models_names[i])

    fig.suptitle('Dimensionality reduction for entire dataset')
    plt.savefig('models.png')


def train_vs_test_plots(x_train, x_test, y_train, y_test):
    isomap_1 = Isomap(n_components=2)
    isomap_2 = Isomap(n_components=2)

    y_train = np.array([int(e[0]) for e in y_train])
    y_test = np.array([int(e[0]) for e in y_test])

    # Fit on train set
    isomap_1.fit(x_train)
    isomap_2.fit(x_test)

    # Transform on train and test
    x_train = isomap_1.transform(x_train)
    x_test = isomap_2.transform(x_test)

    # Plot train
    x_train_df = pd.DataFrame(x_train)
    x_test_df = pd.DataFrame(x_test)
    idx_1 = np.where(y_train == 1)
    idx_0 = np.where(y_train == 0)
    plt.scatter(x_train_df.iloc[idx_1][0], x_train_df.iloc[idx_1][1], s=10, c='b', marker="o", label='old')
    plt.scatter(x_train_df.iloc[idx_0][0], x_train_df.iloc[idx_0][1], s=10, c='r', marker="o", label='young')
    plt.legend()
    plt.title("Train set after Isomap")
    plt.show()
    plt.close()

    # Plot test
    idx_1 = np.where(y_test == 1)
    idx_0 = np.where(y_test == 0)
    plt.scatter(x_test_df.iloc[idx_1][0], x_test_df.iloc[idx_1][1], s=10, c='b', marker="o", label='old')
    plt.scatter(x_test_df.iloc[idx_0][0], x_test_df.iloc[idx_0][1], s=10, c='r', marker="o", label='young')
    plt.title("Test set after Isomap")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    x_train, x_test, y_train, y_test = read()
    x_train, x_test = normalize(x_train, x_test)
    y_train, y_test = to_binary(y_train, y_test)
    x_train, x_test, y_train, y_test = shuffle(x_train, x_test, y_train, y_test)

    x = np.vstack((x_train, x_test))
    y = np.vstack((y_train, y_test))
    y = np.array([int(e[0]) for e in y]) # Each label is tuple of (age, filename). We only need the ages for the
                                         # visualization.

    # isomap(x, y)
    # multidimensional_scaling(x, y)
    # tsne(x, y)
    # pca(x, y)
    plot_all(x, y)
    # train_vs_test_plots(x_train, x_test, y_train, y_test)
