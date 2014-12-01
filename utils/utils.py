__author__ = 'User'
import cv2
import os
import sys
import numpy
import scipy.interpolate
from PIL import Image


def create_curve_func(points):
    """Return a function derived from control points."""
    if points is None:
        return None
    num_points = len(points)
    if num_points < 2:
        return None
    xs, ys = zip(*points)
    if num_points < 4:
        kind = 'linear'
    else:
        kind = 'cubic'  # cubic interpolation need more 4 points
    return scipy.interpolate.interp1d(xs, ys, kind, bounds_error=False)


def create_lookup_array(func, length=256):
    """Return a lookup[ for whole-number inputs to a function."""
    if func is None:
        return None
    lookup_array = numpy.empty(length)
    i = 0
    while i < length:
        func_i = func(i)
        lookup_array[i] = min(max(0, func_i), length - 1)
        i += 1
    return lookup_array


def apply_lookup_array(lookup_array, src, dst):
    """Map a source to a destination using a lookup."""
    if lookup_array is None:
        return
    dst[:] = lookup_array[src]


def create_flat_view(array):
    """Return a 1D view of an array of anny dimensionality.

       Treat image of multi-channels
    """
    flat_view = array.view()
    flat_view.shape = array.size
    return flat_view


def create_composite_func(func0, func1):
    """Return a composite of two functions."""
    if func0 is None:
        return func1
    if func1 is None:
        return func0
    return lambda x: func0(func1(x))


def is_gray(image):
    return image.ndim < 3


def width_height_divided_by(image, divisor):
    h, w = image.shape[:2]
    return w / divisor, h / divisor


def read_images(path, sz=None):
    """Read image in each sub-folder."""
    c = 0
    X, y = [], []
    for dir_name, dir_names, file_names in os.walk(path):
        for subdir_name in dir_names:
            subject_path = os.path.join(dir_name, subdir_name)
            for file_name in os.listdir(subject_path):
                try:
                    im = Image.open(os.path.join(subject_path, file_name))
                    im = im.convert("L")
                    if sz is not None:
                        im = im.resize(sz, Image.ANTIALIAS)
                    X.append(numpy.asarray(im, dtype=numpy.uint8))
                    y.append(c)
                except IOError as err:
                    print "I/O error{0}: {1}".format(err.errno, err.strerror)
                except:
                    print "Unexpected error: ", sys.exc_info()[0]
                    raise
            c += 1
    return [X, y]


def read_image(path):
    im = Image.open(os.path.join(os.getcwd(), path))
    im = im.convert("L")
    return numpy.asarray(im, dtype=numpy.uint8)


def image_to_array(im):
    gray_im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    tmp_path = "tmp.bmp"
    cv2.imwrite(tmp_path, gray_im)
    return read_image(tmp_path)


def as_row_matrix(X):
    if len(X) == 0:
        return numpy.array([])
    mat = numpy.empty((0, X[0].size), dtype=X[0].dtype)
    for row in X:
        mat = numpy.vstack((mat, numpy.asarray(row).reshape(1, -1)))
    return mat


def as_column_matrix(X):
    if len(X) == 0:
        return numpy.array([])
    mat = numpy.empty((X[0].size, 0), dtype=X[0].dtype)
    for col in X:
        mat = numpy.hstack((mat, numpy.asarray(col).reshape(-1, 1)))
    return mat


def normalize(X, low, high, dtype=None):
    """
    Normalize value of X vector to new interval [low, high]
    :param X:
    :param low:
    :param high:
    :param dtype:
    :return:
    """
    X = numpy.asarray(X)
    minX, maxX = numpy.min(X), numpy.max(X)
    X = X - float(minX)
    X = X / float((maxX - minX))
    X = X * (high - low)
    X = X + low
    if dtype is None:
        return numpy.asarray(X)
    return numpy.asarray(X, dtype=dtype)