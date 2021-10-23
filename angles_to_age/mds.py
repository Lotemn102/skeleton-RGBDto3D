import os, sys, getopt, pdb
from numpy import *
from numpy.linalg import *
from numpy.random import *
import pylab
from sklearn.manifold import MDS
from angles_to_age.data_reader import read
from angles_to_age.preprocess import normalize, to_binary, shuffle
import numpy as np

def mds(d, dimensions = 2):
    """
    Multidimensional Scaling - Given a matrix of interpoint distances,
    find a set of low dimensional points that have similar interpoint
    distances.
    """

    (n,n) = d.shape
    E = (-0.5 * d**2)

    # Use mat to get column and row means to act as column and row means.
    Er = mat(mean(E,1))
    Es = mat(mean(E,0))

    # From Principles of Multivariate Analysis: A User's Perspective (page 107).
    F = array(E - transpose(Er) - Es + mean(E))

    [U, S, V] = svd(F)

    Y = U * sqrt(S)

    return (Y[:,0:dimensions], S)

def norm(vec):
    return sqrt(sum(vec**2))

def square_points(size):
    nsensors = size ** 2
    return array([(i / size, i % size) for i in range(nsensors)])

def test():
    x_train, x_test, y_train, y_test = read()
    x_train, x_test = normalize(x_train, x_test)
    y_train, y_test = to_binary(y_train, y_test)
    x_train, x_test, y_train, y_test = shuffle(x_train, x_test, y_train, y_test)

    x = np.vstack((x_train, x_test))
    y = np.vstack((y_train, y_test))
    y = np.array([int(e[0]) for e in y])  # Each label is tuple of (age, filename). We only need the ages for the
    # visualization.

    # Use sklearn to generate the distances matrix
    model = MDS(n_components=3, metric=True)
    model.fit(x)
    D = model.dissimilarity_matrix_
    Y, eigs = mds(D)

    pylab.figure(1)
    pylab.plot(Y[:,0],Y[:,1],'.')

    pylab.figure(2)
    pylab.plot(x[:,0], x[:,1], '.')

    pylab.show()


if __name__ == "__main__":
    test()