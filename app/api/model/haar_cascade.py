import cv2
import numpy as np

_face_detector = cv2.CascadeClassifier()
_face_detector.load('api/model/weights/haarcascade_frontalface_default.xml')

def find_faces(img: np.array) -> list:
    """
        Returns a list of (x, y, w, h) -> bboxes of detected faces
    """
    faces = _face_detector.detectMultiScale(img)
    return faces