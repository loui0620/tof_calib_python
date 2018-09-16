import cv2
import sys
import ctypes

class Camera(object):
    def __init__(self, indexColor = 0, indexDepth = 1):
        self.capColor = cv2.VideoCapture(indexColor)
        self.capDepth = cv2.VideoCapture(indexDepth)
        self.openni = indexColor in (cv2.CAP_OPENNI, cv2.CAP_OPENNI2)
        self.fps = 0
    
    def __enter__(self):
        return self
    
    def __exit(self, exc_type, exc_value, traceback):
        self.release()

    def release(self):
        if not self.capColor: return
        self.capColor.release()
        self.capColor = None
    
    def capture(self, callback, gray=True):
        if not self.capColor:
            sys.exit('The capture is not ready')

        while True:
            t = cv2.getTickCount()

            if self.openni:
                if not self.capColor.grab():
                    sys.exit('Grabs the next frame failed')
                ret, depth = self.cap.retrieve(cv2.CAP_OPENNI_DEPTH_MAP)
                ret, frame = self.cap.retrieve(cv2.CAP_OPENNI_GRAY_IMAGE
                    if gray else cv2.CAP_OPENNI_BGR_IMAGE)
                if callback:
                    callback(frame, depth, self.fps)
            else:
                ret, frame = self.capColor.read()
                if not ret:
                    sys.exit('Reads the next frame failed')
                if gray:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if callback:
                    callback(frame, self.fps)

            t = cv2.getTickCount() - t
            self.fps = cv2.getTickFrequency() / t

            # esc, q
            ch = cv2.waitKey(10) & 0xFF
            if ch == 27 or ch == ord('q'):
                break

