"""
Several algorithms for dimensionality reduction, so i would be able to visualize the data.
"""
import numpy as np
from sklearn.manifold import Isomap, MDS, TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

from angles_to_age.data_reader_2 import read
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


def mds_eigenvals(x):
    # x_ = [e for i, e in enumerate(x) if i %  == 0]
    # x = np.array(x_)

    # Use sklearn to generate the distances matrix
    model = MDS(n_components=3, metric=True)
    model.fit(x)
    D_ = model.dissimilarity_matrix_

    # Number of points
    n = len(D_)

    # Centering matrix
    H = np.eye(n) - np.ones((n, n)) / n

    # YY^T
    B = -H.dot(D_ ** 2).dot(H) / 2

    # Diagonalize
    evals, evecs = np.linalg.eigh(B)

    # Sort by eigenvalue in descending order
    idx = np.argsort(evals)[::-1]
    evals = evals[idx]
    evecs = evecs[:, idx]

    t = np.diag(evals)

    # Compute the coordinates using positive-eigenvalued components only
    w, = np.where(evals > 0)
    L = np.diag(np.sqrt(evals[w]))
    V = evecs[:, w]
    Y = V.dot(L)

    return L, Y


def multidimensional_scaling(x, y, metric=True):
    x = np.array(x)
    y = np.array(y)

    model = MDS(n_components=3, metric=metric)
    proj = model.fit_transform(x)
    proj_df = pd.DataFrame(proj)

    idx_1 = np.where(y == 1)
    idx_0 = np.where(y == 0)

    # 2D Scatter
    # plt.scatter(proj_df.iloc[idx_1][0], proj_df.iloc[idx_1][1], s=10, c='b', marker="o", label='old')
    # plt.scatter(proj_df.iloc[idx_0][0], proj_df.iloc[idx_0][1], s=10, c='r', marker="o", label='young')

    # 3D Scatter
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(proj_df.iloc[idx_1][0], proj_df.iloc[idx_1][1], proj_df.iloc[idx_1][2], s=10, c='b', marker="o", label='old')
    ax.scatter(proj_df.iloc[idx_0][0], proj_df.iloc[idx_0][1], proj_df.iloc[idx_0][2], s=10, c='r', marker="o", label='young')

    plt.legend()
    plt.show()

    return proj


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
    # x_train, x_test = normalize(x_train, x_test)
    y_train, y_test = to_binary(y_train, y_test)
    x_train, x_test, y_train, y_test = shuffle(x_train, x_test, y_train, y_test)

    x = np.vstack((x_train, x_test))
    y = np.vstack((y_train, y_test))
    y = np.array([int(e[0]) for e in y]) # Each label is tuple of (age, filename). We only need the ages for the
                                         # visualization.

    # isomap(x, y)
    K = multidimensional_scaling(x, y)
    # tsne(x, y)
    # pca(x, y)
    # plot_all(x, y)
    # train_vs_test_plots(x_train, x_test, y_train, y_test)
    # L, Y = mds_eigenvals(x)

    t = 4

