__author__ = 'Sparrow'
import cv2
from utils import rects, utils


class Face(object):
    """Data on facial features: face, eyes, nose, mouth."""
    def __init__(self):
        self.faceRect = None
        self.leftEyeRect = None
        self.rightEyeRect = None
        self.noseRect = None
        self.mouthRect = None


class FaceTracker(object):
    """A tracker for facial features: face, eyes, nose, mouth."""
    def __init__(self, scale_factor = 1.2, min_neighbors = 2,
                 flags = cv2.CASCADE_SCALE_IMAGE):
        self.scaleFactor = scale_factor
        self.minNeighbors = min_neighbors
        self.flags = flags
        self._faces = []
        # print(os.path.abspath('haarcascade_frontalface_alt.xml'))
        self._faceClassifier = cv2.CascadeClassifier(
            'cascades/haarcascade_frontalface_alt.xml')
        self._eyeClassifier = cv2.CascadeClassifier(
            'cascades/haarcascade_eye.xml')
        self._noseClassifier = cv2.CascadeClassifier(
            'cascades/haarcascade_mcs_nose.xml')
        self._mouthClassifier = cv2.CascadeClassifier(
            'cascades/haarcascade_mcs_mouth.xml')

    @property
    def faces(self):
        """The tracked facial features."""
        return self._faces

    def update(self, image):
        """Update the tracked facial features."""
        self._faces = []
        if utils.is_gray(image):
            image = cv2.equalizeHist(image)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            cv2.equalizeHist(image, image)
        min_size = utils.width_height_divided_by(image, 8)
        face_rect = self._faceClassifier.detectMultiScale(
            image, self.scaleFactor, self.minNeighbors, self.flags,
            min_size)
        if face_rect is not None:
            for faceRect in face_rect:
                face = Face()
                face.faceRect = faceRect
                x, y, w, h = faceRect
                search_rect = (x+w/7, y, w*2/7, h/2)
                face.leftEyeRect = self._detect_one_object(
                    self._eyeClassifier, image, search_rect, 64)

                search_rect = (x+w*4/7, y, w*2/7, h/2)
                face.rightEyeRect = self._detect_one_object(
                    self._eyeClassifier, image, search_rect, 64)

                search_rect = (x+w/4, y + h/4, w/2, h/2)
                face.noseRect = self._detect_one_object(
                    self._noseClassifier, image, search_rect, 32)

                search_rect = (x+w/6, y + h*2/3, w*2/3, h/3)
                face.mouthRect = self._detect_one_object(
                    self._mouthClassifier, image, search_rect, 16)

                self._faces.append(face)

    def _detect_one_object(self, classifier, image, rect,
                           image_size_to_min_size_ratio):
        x, y, w, h = rect
        min_size = utils.width_height_divided_by(
            image, image_size_to_min_size_ratio)
        sub_image = image[y:y+h, x:x+w]
        sub_rect = classifier.detectMultiScale(
            sub_image, self.scaleFactor, self.minNeighbors,
            self.flags, min_size)
        if len(sub_rect) == 0:
            return None
        sub_x, sub_y, sub_w, sub_h = sub_rect[0]
        return x + sub_x, y + sub_y, sub_w, sub_h

    def draw_debug_rect(self, image):
        """Draw rectangles around the tracked facial features."""
        if utils.is_gray(image):
            face_color = 255
            left_eye_color = 255
            right_eye_color = 255
            nose_color = 255
            mouth_color = 255
        else:
            face_color = (255, 255, 255)
            left_eye_color = (0, 255, 255)
            right_eye_color = (0, 0, 255)
            nose_color = (0, 255, 0)
            mouth_color = (0, 0, 0)
        for face in self.faces:
            rects.outline_rect(image, face.faceRect, face_color)
            rects.outline_rect(image, face.leftEyeRect, left_eye_color)
            rects.outline_rect(image, face.rightEyeRect, right_eye_color)
            rects.outline_rect(image, face.noseRect, nose_color)
            rects.outline_rect(image, face.mouthRect, mouth_color)
