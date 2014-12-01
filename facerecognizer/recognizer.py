__author__ = 'User'
import cv2
from cv2 import __version__
from utils.utils import normalize, as_column_matrix, as_row_matrix, read_images, read_image, image_to_array
from utils.visual import subplot
from subspace import pca, project, reconstruct
import matplotlib.cm as cm
from model import EigenfacesModel


class FaceRecognizer(object):
    def __init__(self):
        self._facePath = "orl_faces"

    def visualize(self):
        # Using PCA (Principal Components Analysis) to recognize faces
        # Face recognizer just in openCV version 2.4x, not in version 3.0 (alpha + beta)
        # cv2.createEigenFaceRecognizer()
        [X, y] = read_images(self._facePath)
        # get k principal components
        [D, W, mu] = pca(as_row_matrix(X), y)
        # to check how many principal components (eigenfaces) - which has the same dimension with observed vector
        # print(len(W[1, :]))
        # print(len(W[:, 1]))

        # Visualize eigenfaces
        E = []
        # take at most 16 people
        for i in xrange(min(len(X), 16)):
            e = W[:, i].reshape(X[0].shape)
            # normalize value of eigenvectors in [0, 255] to display on image
            E.append(normalize(e, 0, 255))
        # display k principal components (k eigenvectors)
        # eigenfaces encode facial features + illumination of image
        subplot(title="Eigencefaces AT&T", images=E, rows=4, cols=4, sptitle="Eigenface", colormap=cm.gray,
                filename="pca_eigenface.png")

        # Reconstructor image from eigenfaces
        # array of vector number [10, 30, 50 ..., min(len(X), 320)]
        steps = [i for i in xrange(10, min(len(X), 320), 20)]
        E = []
        # take at most 16 eigenvectors of first person
        for i in xrange(min(len(steps), 16)):
            num_eigenvectors = steps[i]
            # project each vector x1, x2, ... xn to num_eigenvectors
            P = project(W[:, 0:num_eigenvectors], X[0].reshape(1, -1), mu)
            # reconstruct image from num_eigenvectors
            R = reconstruct(W[:, 0:num_eigenvectors], P, mu)
            R = R.reshape(X[0].shape)
            E.append(normalize(R, 0, 255))
        # reconstruct image from k (10, 30, ...) eigenvector
        subplot(title="Reconstruction AT&T", images=E, rows=4, cols=4, sptitle="EigenVectors", sptitles=steps,
                colormap=cm.gray, filename="pca_reconstruction.png")

    def recognize(self, frame):
        [X, y] = read_images(self._facePath)
        # Recognize face
        model = EigenfacesModel(X[1:], y[1:])
        hits = 0
        for i in xrange(len(X)):
            ret = model.predict(X[i])
            print "Expected =", y[i], "/", "predict =", ret
            hits += (ret == y[i])
        print "Total hit: ", hits, " over ", len(X), " observations"

        obj_name = "Sparrow"
        # Just for test
        # obj = image_to_array(frame)
        # ret = model.predict(obj)
        # # print "Expected = mobile(0) /", "predict =", ret
        # obj_name = "Mobile" if (ret == 0) else "Cup"
        return obj_name