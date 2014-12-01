__author__ = 'User'
import numpy as np
from utils.utils import as_row_matrix
from subspace import pca, project
from distance import EuclideanDistance, CosineDistance


class BaseModel(object):
    def __init__(self, X=None, y=None, dist_metric=EuclideanDistance(), num_components=0):
        self.dist_metric = dist_metric
        self.num_components = num_components
        self.projections = []
        self.W = []
        self.mu = []
        if (X is not None) and (y is not None):
            self.compute(X, y)

    def compute(self):
        raise NotImplementedError("Every BaseModel must implement the compute method")

    def predict(self, X):
        min_dist = np.finfo('float').max
        min_class = -1
        Q = project(self.W, X.reshape(1, -1), self.mu)
        for i in xrange(len(self.projections)):
            dist = self.dist_metric(self.projections[i], Q)
            if dist < min_dist:
                min_dist = dist
                min_class = self.y[i]
        return min_class


class EigenfacesModel(BaseModel):
    def __init__(self, X=None, y=None, dist_metric=EuclideanDistance(), num_components=0):
        super(EigenfacesModel, self).__init__(X=X, y=y, dist_metric=dist_metric, num_components=num_components)

    def compute(self, X, y):
        [D, self.W, self.mu] = pca(as_row_matrix(X), y, self.num_components)
        self.y = y
        for xi in X:
            self.projections.append(project(self.W, xi.reshape(1, -1), self.mu))
