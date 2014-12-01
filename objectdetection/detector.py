__author__ = 'User'
import cv2


class Model(object):
    def __init__(self, key_point_type="FAST"):
        # ["FAST","STAR","SIFT","SURF","ORB","MSER","GFTT","HARRIS"]
        self.key_point_type = key_point_type


class ObjectDetector(object):
    def __init__(self):
        self.model = Model(key_point_type="FAST")
        self._imagePath = "Cup.bmp"

    def display_key_points(self):
        im = cv2.imread(self._imagePath)
        gray_im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        detector = cv2.FeatureDetector_create(self.model.key_point_type)
        key_points = detector.detect(gray_im)
        # descriptor_extractor = cv2.DescriptorExtractor_create(self.model.key_point_type)
        # (key_points, descriptors) = descriptor_extractor.compute(gray_im, key_points)
        cv2.drawKeypoints(gray_im, key_points, im)
        cv2.imwrite("Image with keypoints.bmp", im)