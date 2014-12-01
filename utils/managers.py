__author__ = 'User'
import cv2
import numpy
import time
import pygame
import utils


class CaptureManager(object):
    def __init__(self, capture, preview_window_manager=None,
                 should_mirror_preview=False):
        # '__' stand for private
        self.previewWindowManager = preview_window_manager
        self.shouldMirrorPreview = should_mirror_preview
        self._capture = capture  # '_' stand for protect
        self._channel = 0
        self._enteredFrame = False
        self._frame = None
        self._imageFilename = None
        self._videoFilename = None
        self._videoEncoding = None
        self._videoWriter = None
        self._startTime = None
        self._framesElapsed = long(0)
        self._fpsEstimate = None

    @property
    def channel(self):
        return self._channel

    @channel.setter
    def channel(self, value):
        if self._channel != value:
            self._channel = value
            self._frame = None

    @property
    def frame(self):
        if self._enteredFrame and self._frame is None:
            _, self._frame = self._capture.retrieve(self._channel)
        return self._frame

    @property
    def is_writing_image(self):
        return self._imageFilename is not None

    @property
    def is_writing_video(self):
        return self._videoFilename is not None

    def enter_frame(self):
        """Capture the next frame, if any."""
        # First, check that any previous frame was exited.
        assert not self._enteredFrame, 'previous enterFrame() has no matching exitFrame()'
        if self._capture is not None:
            self._enteredFrame = self._capture.grab()

    def exit_frame(self):
        """Draw to windows, write to files, release the frame."""
        # Check whether any grabbed frame is retrievable.
        # The getter may retrieve and cache the frame.
        if self.frame is None:
            self._enteredFrame = False
            return
        # Update the FPS estimate and related variables.
        if self._framesElapsed == 0:
            self._startTime = time.time()
        else:
            time_elapsed = time.time() - self._startTime
            self._fpsEstimate = self._framesElapsed / time_elapsed
        self._framesElapsed += 1
        # Draw to windows, if any.
        if self.previewWindowManager is not None:
            if self.shouldMirrorPreview:
                mirrored_frame = numpy.fliplr(self._frame).copy()
                self.previewWindowManager.show(mirrored_frame)
            else:
                self.previewWindowManager.show(self._frame)
        # Write to the image file, if any.
        if self.is_writing_image:
            cv2.imwrite(self._imageFilename, self._frame)
            self._imageFilename = None
        # Write to the video file, if any.
        if self.is_writing_video:
            self._write_video_frame()
        # Release frame
        self._frame = None
        self._enteredFrame = False

    def write_image(self, filename):
        """Write the next exited frame to an image file."""
        self._imageFilename = filename

    def start_writing_video(self, filename, encoding=cv2.VideoWriter_fourcc('I', '4', '2', '0')):
        """Start writing exited frames to a video file."""
        self._videoFilename = filename
        self._videoEncoding = encoding

    def stop_writing_video(self):
        """Stop writing exited frames to a video file."""
        self._videoFilename = None
        self._videoEncoding = None
        self._videoWriter = None

    def _write_video_frame(self):
        if not self.is_writing_video:
            return
        if self._videoWriter is None:
            fps = self._capture.get(cv2.CAP_PROP_FPS)
            if fps == 0.0:
                # The capture FPS is unknown so use an estimate.
                if self._framesElapsed < 20:
                    # Wait until more frames elapse so that
                    # the estimate is more stable
                    return
                else:
                    fps = self._fpsEstimate
            size = (int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            self._videoWriter = cv2.VideoWriter(self._videoFilename,
                                                self._videoEncoding, fps, size)
        self._videoWriter.write(self._frame)


class WindowManager(object):
    def __init__(self, window_name, keypress_callback=None):
        self.keypressCallback = keypress_callback
        self._windowName = window_name
        self._isWindowCreated = False

    @property
    def is_window_created(self):
        return self._isWindowCreated

    def create_window(self):
        cv2.namedWindow(self._windowName)
        self._isWindowCreated = True

    def show(self, frame):
        cv2.imshow(self._windowName, frame)

    def destroy_window(self):
        cv2.destroyWindow(self._windowName)
        self._isWindowCreated = False

    def process_events(self):
        key_code = cv2.waitKey(1)
        if self.keypressCallback is not None and key_code != -1:
            # Discard any non-ASCII info encoded by GTK.
            key_code &= 0xFF
            self.keypressCallback(key_code)


class PygWindowManager(WindowManager):
    def create_window(self):
        pygame.init()
        pygame.display.init()
        pygame.display.set_caption(self._windowName)
        self._isWindowCreated = True

    def show(self, frame):
        frame_size = frame.shape[1::-1]
        if utils.is_gray(frame):
            conversion_type = cv2.COLOR_GRAY2RGB
        else:
            conversion_type = cv2.COLOR_BGR2RGB
        rgb_frame = cv2.cvtColor(frame, conversion_type)
        pyg_frame = pygame.image.frombuffer(
            rgb_frame.tostring(), frame_size, 'RGB')
        display_surface = pygame.display.set_mode(frame_size)
        display_surface.blit(pyg_frame, (0, 0))
        pygame.display.flip()

    def destroy_window(self):
        pygame.display.quit()
        self._isWindowCreated = False

    def process_events(self):
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN and \
               self.keypressCallback is not None:
                self.keypressCallback(event.key)
            elif event.type == pygame.QUIT:
                self.destroy_window()
                return

    def display_text(self, name):
        pygame.display.set_caption(name)
