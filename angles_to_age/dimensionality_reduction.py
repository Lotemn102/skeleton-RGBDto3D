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

def pca(x, y, sub_nums):
    model = PCA(n_components=2)
    model = model.fit(x)

    first_comp = model.components_[0]
    second_comp = model.components_[1]

    print(np.mean(first_comp))
    print(np.mean(second_comp))
    x_proj = np.dot(x, first_comp)
    y_proj = np.dot(x, second_comp)

    # Weights of first pc
    indices = range(1, len(first_comp)+1)
    #indices = ["alpha_" + str(i) for i in range(1, len(first_comp)+1)]
    stacked = np.vstack((indices, first_comp)).T
    stacked = pd.DataFrame(stacked)
    stacked.columns = ['alpha', 'PC1']
    print(stacked)

    # Separation by first pc
    idx_1 = np.where(y == 1)
    idx_0 = np.where(y == 0)
    a = x_proj[idx_1]
    b = x_proj[idx_0]
    plt.scatter(a, [0]*len(a), s=10, c='r')
    plt.scatter(b, [0]*(b), s=10)
    plt.show()
    plt.close()

    # Separation by second pc
    idx_1 = np.where(y == 1)
    idx_0 = np.where(y == 0)
    a = y_proj[idx_1]
    b = y_proj[idx_0]
    plt.scatter(a, [0] * len(a), s=10, c='r')
    plt.scatter(b, [0] * (b), s=10)
    plt.show()
    plt.close()

    # Plot projection by both axes
    proj = model.transform(x)
    proj_df = pd.DataFrame(proj)

    idx_1 = np.where(y == 1)
    idx_0 = np.where(y == 0)

    # Plot with subject numbers
    fig, ax = plt.subplots()
    ax.scatter(proj_df.iloc[idx_1][0], proj_df.iloc[idx_1][1], s=80, c='b', marker="o", label='old')
    ax.scatter(proj_df.iloc[idx_0][0], proj_df.iloc[idx_0][1], s=80, c='r', marker="o", label='young')

    x_temp = list(proj_df.iloc[idx_1][0])
    y_temp = list(proj_df.iloc[idx_1][1])
    sub_nums_1 = sub_nums[idx_1]
    sub_nums_0 = sub_nums[idx_0]

    for i, txt in enumerate(sub_nums_1):
        plt.text(x_temp[i] - 0.01, y_temp[i] - 0.12, str(txt), fontsize=9)

    x_temp = list(proj_df.iloc[idx_0][0])
    y_temp = list(proj_df.iloc[idx_0][1])
    for i, txt in enumerate(sub_nums_0):
        plt.text(x_temp[i] - 0.01, y_temp[i] - 0.12, str(txt), fontsize=9)

    plt.title("PCA, with subject number")
    plt.legend()
    plt.savefig('pca_numbers.png')

    # Plot with subject age
    NUM_TO_AGE_MAP = {1: 24, 2: 26, 3: 27, 4: 27, 5: 27, 6: 25, 7: 27, 8: 30, 9: 23, 10: 27, 11: 26, 12: 27, 13: 23,
                      14: 26, 15: 26, 16: 27, 17: 85, 18: 72, 19: 77, 20: 72, 21: 66, 22: 74, 23: 67}
    ages = np.array([NUM_TO_AGE_MAP[i] for i in sub_nums])

    fig, ax = plt.subplots()
    ax.scatter(proj_df.iloc[idx_1][0], proj_df.iloc[idx_1][1], s=200, c='b', marker="o", label='old')
    ax.scatter(proj_df.iloc[idx_0][0], proj_df.iloc[idx_0][1], s=200, c='r', marker="o", label='young')

    x_temp = list(proj_df.iloc[idx_1][0])
    y_temp = list(proj_df.iloc[idx_1][1])
    ages_nums_1 = ages[idx_1]
    ages_nums_0 = ages[idx_0]

    for i, txt in enumerate(ages_nums_1):
        plt.text(x_temp[i] - 0.025, y_temp[i] - 0.015, str(txt), fontsize=9, color="white")

    x_temp = list(proj_df.iloc[idx_0][0])
    y_temp = list(proj_df.iloc[idx_0][1])
    for i, txt in enumerate(ages_nums_0):
        plt.text(x_temp[i] - 0.025, y_temp[i] - 0.015, str(txt), fontsize=9, color="white")

    plt.title("PCA, with subject age")
    plt.legend()
    plt.savefig('pca_ages.png')


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
    # x_train, x_test, y_train, y_test = shuffle(x_train, x_test, y_train, y_test)

    x = np.vstack((x_train, x_test))
    y = np.vstack((y_train, y_test))
    sub_nums = np.array([int(e[1][4:][:-10]) for e in y])
    y = np.array([int(e[0]) for e in y]) # Each label is tuple of (age, filename). We only need the ages for the
                                         # visualization.

    # isomap(x, y)
    # multidimensional_scaling(x, y)
    # tsne(x, y)
    pca(x, y, sub_nums)
    # plot_all(x, y)
    # train_vs_test_plots(x_train, x_test, y_train, y_test)
    # L, Y = mds_eigenvals(x)

    t = 4

