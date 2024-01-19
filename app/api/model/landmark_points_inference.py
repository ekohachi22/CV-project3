import cv2

import tensorflow as tf
import numpy as np
from tensorflow import keras as keras

_model = keras.models.load_model("api/model/weights/landmark.keras")
INPUT_SIZE = (288, 288, 1)

def infer(img: np.array) -> list:
    if img.shape != INPUT_SIZE:
        img = cv2.resize(img, INPUT_SIZE)
    img /= 255
    return _model.predict(np.array([img]))
