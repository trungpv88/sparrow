__author__ = 'User'
import numpy as np
import cv2


class BaseModel(object):
    def __init__(self, name=None):
        self.name = name

    def compute(self):
        raise NotImplementedError("Must implement compute method.")


class WaterShed(BaseModel):
    def __init__(self, img=None):
        super(WaterShed, self).__init__(name="watershed")
        self.image = img

    def compute(self):
        gray_im = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray_im, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        # noise removal
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        # find sure background
        sure_background = cv2.dilate(opening, kernel, iterations=3)
        # find sure foreground
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        ret, sure_foreground = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
        # find unknown region
        sure_foreground = np.uint8(sure_foreground)
        unknown = cv2.subtract(sure_background, sure_foreground)
        # marker labelling
        ret, markers = cv2.connectedComponents(sure_foreground)
        # add 1 to all labels so that sure background is not 0, but 1
        markers += 1
        # mark unknown region with 0
        markers[unknown == 255] = 0
        # watershed
        markers = cv2.watershed(self.image, markers)
        self.image[markers == -1] = [0, 255, 0]
        # cv2.imwrite("xxxxxxxxxxxxxxxx.png", self.image)