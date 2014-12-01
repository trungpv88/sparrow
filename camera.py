__author__ = 'Sparrow'
import cv2
import os
from utils import filters, utils
from utils.managers import PygWindowManager as WindowManager, CaptureManager
from facetracking.trackers import FaceTracker
from facerecognizer.recognizer import FaceRecognizer
from objectdetection.detector import ObjectDetector
from segmentation.model import WaterShed
from reconstruction3D import Reconstruction3D


class Camera(object):
    def __init__(self):
        self._windowManager = WindowManager('Sparrow', self.on_key_press)
        self._captureManager = CaptureManager(cv2.VideoCapture(0),
                                              self._windowManager, False)
        self._faceTracker = FaceTracker()
        self._shouldDrawDebugRect = False
        self._curveFilter = filters.BGRFujiCurveFilter()
        self._faceRecognizer = FaceRecognizer()
        self._visualizeFace = False
        self._recognizeFace = False
        self._detectObject = False
        self._objDetector = ObjectDetector()
        self._segmentation = False
        self._watershed = None
        self._reconstruction3D = None
        self._object3D = None
        self._num_img_captured = 0

    def run(self):
        """Run main loop."""
        # utils.read_images(path=os.path.join(os.getcwd(), "orl_faces"))
        self._windowManager.create_window()
        while self._windowManager.is_window_created:
            self._captureManager.enter_frame()
            frame = self._captureManager.frame

            self._faceTracker.update(frame)
            # faces = self._faceTracker.faces
            # rects.swapRects(frame, frame, [face.faceRect for face in faces])
            #
            # # TODO: Filter the frame
            # filters.strokeEdges(frame, frame)
            # self._curveFilter.apply(frame, frame)

            if self._shouldDrawDebugRect:
                self._faceTracker.draw_debug_rect(frame)

            if self._visualizeFace:
                self._faceRecognizer.visualize()
                self._visualizeFace = False

            if self._recognizeFace:
                obj_name = self._faceRecognizer.recognize(frame)
                self._windowManager.display_text(obj_name)
                self._recognizeFace = False

            if self._detectObject:
                self._objDetector.display_key_points()
                self._detectObject = False

            if self._segmentation:
                # image = cv2.imread("NB041.jpg")
                self._watershed = WaterShed(img=frame)
                # self._watershed = WaterShed(img=image)
                self._watershed.compute()
                # self._segmentation = False

            if self._reconstruction3D:
                self._num_img_captured += 1
                if self._num_img_captured == 1:
                    frame_left = frame
                if self._num_img_captured == 2:
                    self._object3D = Reconstruction3D(frame_left, frame)
                    self._object3D.reconstruct()
                    self._num_img_captured = 0
                self._reconstruction3D = False

            self._captureManager.exit_frame()
            self._windowManager.process_events()

    def on_key_press(self, keycode):
        """Handle a key press:

        space  -> take a screen shot.
        tab    -> Start/Stop recording a screen cast.
        escape -> Quit.
        x      -> Start/Stop drawing debug rectangle around faces.
        o      -> Detect object..
        r      -> Recognize face
        s      -> Segmentation.
        t      -> 3D reconstruction
        v      -> Visualise face.

        """
        if keycode == 32:  # space
            self._captureManager.write_image('screenshot.png')
        elif keycode == 9:  # tab
            if not self._captureManager.is_writing_video:
                self._captureManager.start_writing_video('screencast.avi')
            else:
                self._captureManager.stop_writing_video()
        elif keycode == 120:  # x
            self._shouldDrawDebugRect = not self._shouldDrawDebugRect
        elif keycode == 111:  # o
            self._detectObject = True
        elif keycode == 114:  # r
            self._recognizeFace = True
        elif keycode == 115:  # s
            self._segmentation = True
        elif keycode == 116:  # t
            self._reconstruction3D = True
        elif keycode == 118:  # v
            self._visualizeFace = True
        elif keycode == 27:  # escape
            self._windowManager.destroy_window()

if __name__ == "__main__":
    Camera().run()

