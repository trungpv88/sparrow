__author__ = 'User'
import cv2
from matplotlib import pyplot as plt


class Reconstruction3D(object):
    def __init__(self, image_left=None, image_right=None):
        self.image_left = cv2.cvtColor(image_left, cv2.COLOR_BGR2GRAY)
        self.image_right = cv2.cvtColor(image_right, cv2.COLOR_BGR2GRAY)
        # self.image_left = cv2.imread('aloeL.jpg', cv2.CV_8UC1)
        # self.image_right = cv2.imread('aloeR.jpg', cv2.CV_8UC1)

    def reconstruct(self):
        stereo = cv2.createStereoBM(numDisparities=32, blockSize=9)
        disparity = stereo.compute(self.image_left, self.image_right)
        plt.imshow(disparity, 'gray')
        plt.show()