import cv2

import tensorflow as tf
import numpy as np
from tensorflow import keras as keras

_model = keras.models.load_model("api/model/weights/model4_Adam_mean_squared_error.keras", safe_mode = False)
INPUT_SIZE = (96, 96, 1)

def infer(img: np.array) -> list:
    if img.shape != INPUT_SIZE:
        img = cv2.resize(img, INPUT_SIZE[:2])
    return _model.predict(np.array([img]))
