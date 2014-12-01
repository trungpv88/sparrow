__author__ = 'User'
import numpy as np


def pca(X, y, num_components=0):
    # get [Number of observations, dimension] of image array
    [n, d] = X.shape
    # by default, number of principal components = number of observations
    if (num_components <= 0) or (num_components > n):
        num_components = n
    # compute the mean of random vector X = {x1, x2, ..., xn} (matrix of n x 1)
    mu = X.mean(axis=0)
    X = X - mu
    if n > d:
        # compute covariance matrix n x n: S = 1/n sum (x_i - mu)(x_i - mu)T
        covariance_mat = np.dot(X.T, X)
        # numpy.linalg.eigh return eigenvalues and eigenvectors of symmetric matrix: S * v_i = lamda_i * v_i
        [eigenvalues, eigenvectors] = np.linalg.eigh(covariance_mat)
    else:
        covariance_mat = np.dot(X, X.T)
        [eigenvalues, eigenvectors] = np.linalg.eigh(covariance_mat)
        eigenvectors = np.dot(X.T, eigenvectors)
        for i in xrange(n):
            eigenvectors[:, i] = eigenvectors[:, i] / np.linalg.norm(eigenvectors[:, i])
    # get indices of eigenvectors descending by eigenvalue
    idx = np.argsort(-eigenvalues)
    # sort eigenvalues and eigenvector with new order
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    # take k principal components (k largest eigenvalues)
    eigenvalues = eigenvalues[0: num_components].copy()
    eigenvectors = eigenvectors[:, 0:num_components].copy()
    return [eigenvalues, eigenvectors, mu]


def project(W, X, mu=None):
    if mu is None:
        return np.dot(X, W)
    # y = W^T(X - mu)
    return np.dot(X - mu, W)


def reconstruct(W, Y, mu=None):
    """
    Reconstruct image from eigenvectors W
    :param W:
    :param Y:
    :param mu:
    :return:
    """
    if mu is None:
        return np.dot(Y, W.T)
    # x = Y * W^T + mu
    return np.dot(Y, W.T) + mu


