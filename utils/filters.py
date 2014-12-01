__author__ = 'User'
import cv2
import numpy
import utils


def recolor_rc(src, dst):
    """Simulate conversion from BGR to RC (red, cyan).

       (b, g, r) -> (0.5b + 0.5g, 0.5b + 0.5g, r)
    """
    b, g, r = cv2.split(src)
    cv2.addWeighted(b, 0.5, g, 0.5, b)
    cv2.merge((b, b, r), dst)


def recolor_rgv(src, dst):
    """Simulate conversion from BGR to RGV (red, green, value).

       (b, g, r) -> ((min(b, g, r), g, r)
    """
    b, g, r = cv2.split(src)
    cv2.min(b, g, b)  # b = min(g, b)
    cv2.min(b, r, b)
    cv2.merge((b, g, r), dst)


def recolor_cmv(src, dst):
    """Simulate conversion from BGR to RGV (cyan, magenta, value).

       (b, g, r) -> ((max(b, g, r), g, r)
    """
    b, g, r = cv2.split(src)
    cv2.max(b, g, b)  # b = min(g, b)
    cv2.max(b, r, b)
    cv2.merge((b, g, r), dst)


class VFuncFilter(object):
    """A filter that applies a function to V (or all of BGR)."""
    def __init___(self, v_func=None, d_type=numpy.uint8):
        length = numpy.iinfo(d_type).max + 1
        self._vLookupArray = utils.createdLookupArray(v_func, length)

    def apply(self, src, dst):
        """Apply the filter with a BGR or gray src/dst"""
        src_flat_view = utils.flatView(src)
        dst_flat_view = utils.flatView(dst)
        utils.apply_lookup_array(self._vLookupArray, src_flat_view, dst_flat_view)


class VCurveFilter(VFuncFilter):
    """A filter that applies a curve to V (or all of BGR)."""
    def __init__(self, v_points, d_type=numpy.uint8):
        VFuncFilter.__init__(self, utils.createCurveFUnc(v_points), d_type)


class BGRFuncFilter(object):
    """A filter that applies different functions to each of BGR."""
    def __init__(self, v_func=None, b_func=None, g_func=None,
                 r_func=None, d_type=numpy.uint8):
        length = numpy.iinfo(d_type).max + 1
        self._bLookupArray = utils.create_lookup_array(
            utils.create_composite_func(b_func, v_func), length)
        self._gLookupArray = utils.create_lookup_array(
            utils.create_composite_func(g_func, v_func), length)
        self._rLookupArray = utils.create_lookup_array(
            utils.create_composite_func(r_func, v_func), length)

    def apply(self, src, dst):
        """Apply the filter with a BGR source/destination."""
        b, g, r = cv2.split(src)
        utils.apply_lookup_array(self._bLookupArray, b, b)
        utils.apply_lookup_array(self._gLookupArray, g, g)
        utils.apply_lookup_array(self._rLookupArray, r, r)
        cv2.merge([b, g, r], dst)


class BGRCurveFilter(BGRFuncFilter):
    """A filter that applies different curves to each of BGR."""
    def __init__(self, v_points=None, b_points=None,
                 g_points=None, r_points=None, d_type=numpy.uint8):
        BGRFuncFilter.__init__(self,
                               utils.create_curve_func(v_points),
                               utils.create_curve_func(b_points),
                               utils.create_curve_func(g_points),
                               utils.create_curve_func(r_points), d_type)


class BGRPortraCurveFilter(BGRCurveFilter):
    """A filter that applies Portra-like curves to BGR."""
    def __init__(self, d_type=numpy.uint8):
        BGRCurveFilter.__init__(
            self, d_type=d_type,
            v_points=[(0, 0), (23, 20), (157, 173), (255, 255)],
            b_points=[(0, 0), (41, 46), (231, 228), (255, 255)],
            g_points=[(0, 0), (52, 47), (189, 196), (255, 255)],
            r_points=[(0, 0), (69, 69), (213, 218), (255, 255)])


class BGRProviaCurveFilter(BGRCurveFilter):
    """A filter that applies Provia-like curves to BGR."""
    def __init__(self, d_type = numpy.uint8):
        BGRCurveFilter.__init__(
            self,
            b_points=[(0, 0), (35, 35), (205, 227), (255, 255)],
            g_points=[(0, 0), (27, 21), (196, 207), (255, 255)],
            r_points=[(0, 0), (59, 54), (202, 210), (255, 255)],
            d_type=d_type)


class BGRFujiCurveFilter(BGRCurveFilter):
    """A filter that applies Velvia-like curves to BGR."""
    def __init__(self, d_type = numpy.uint8):
        BGRCurveFilter.__init__(
            self,
            v_points=[(0, 0), (128, 118), (221, 215), (255, 255)],
            b_points=[(0, 0), (25, 21), (122, 153), (165, 206), (255, 255)],
            g_points=[(0, 0), (25, 21), (95, 102), (181, 208), (255, 255)],
            r_points=[(0, 0), (41, 28), (183, 209), (255, 255)],
            d_type=d_type)

class BGRCrossProcessCurveFilter(BGRCurveFilter):
    """A filter that applies cross-process-like curves to BGR."""
    def __init__(self, d_type=numpy.uint8):
        BGRCurveFilter.__init__(
            self,
            b_points=[(0, 20), (255, 235)],
            g_points=[(0, 0), (56, 39), (208, 226), (255, 255)],
            r_points=[(0, 0), (56, 22), (211, 255), (255, 255)],
            d_type=d_type)


def stroke_edges(src, dst, blur_k_size=7, edge_k_size=5):
    if blur_k_size >= 3:
        blurred_src = cv2.medianBlur(src, blur_k_size)
        gray_src = cv2.cvtColor(blurred_src, cv2.COLOR_BGR2GRAY)
    else:
        gray_src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    cv2.Laplacian(gray_src, cv2.CV_8U, gray_src, ksize=edge_k_size)
    normalized_inverse_alpha = (1.0 / 255) * (255 - gray_src)
    channels = cv2.split(src)
    for channel in channels:
        channel[:] = channel * normalized_inverse_alpha
    cv2.merge(channels, dst)


class VConvolutionFilter(object):
    """A filter that applies a convolution to V (or all of BGR)."""
    def __init__(self, kernel):
        self._kernel = kernel

    def apply(self, src, dst):
        """Apply the filter with a BGR or gray source/destination."""
        cv2.filter2D(src, -1, self._kernel, dst)


class SharpenFilter(VConvolutionFilter):
    """A sharpen filter with a 1-pixel radius."""
    def __init__(self):
        kernel = numpy.array([[-1, -1, -1],
                              [-1, 9, -1],
                              [-1, -1, -1]])
        VConvolutionFilter.__init__(self, kernel)


class FindEdgesFilter(VConvolutionFilter):
    """A edge-finding filter with a 1-pixel radius."""
    def __init__(self):
        kernel = numpy.array([[-1, -1, -1],
                              [-1, 8, -1],
                              [-1, -1, -1]])
        VConvolutionFilter.__init__(self, kernel)


class BlurFilter(VConvolutionFilter):
    """A blur filter with a 1-pixel radius."""
    def __init__(self):
        kernel = numpy.array([[0.04, 0.04, 0.04, 0.04, 0.04],
                              [0.04, 0.04, 0.04, 0.04, 0.04],
                              [0.04, 0.04, 0.04, 0.04, 0.04],
                              [0.04, 0.04, 0.04, 0.04, 0.04],
                              [0.04, 0.04, 0.04, 0.04, 0.04]])
        VConvolutionFilter.__init__(self, kernel)


class EmbossFilter(VConvolutionFilter):
    """A emboss filter with a 1-pixel radius."""
    def __init__(self):
        kernel = numpy.array([[-2, -1, 0],
                              [-1, 1, 1],
                              [0, 1, 2]])
        VConvolutionFilter.__init__(self, kernel)
