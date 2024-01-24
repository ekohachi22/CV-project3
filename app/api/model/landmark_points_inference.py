import cv2

import onnxruntime as ort
import numpy as np

_model = ort.InferenceSession("api/model/weights/model3_Adam_mean_squared_error.onnx")
INPUT_SIZE = (96, 96, 1)

def infer(img: np.array) -> list:
    if img.shape != INPUT_SIZE:
        img = cv2.resize(img, INPUT_SIZE[:2])
    input = np.array([img], dtype=np.float32)
    return _model.run(None, {"input": input.reshape((1,)+INPUT_SIZE)})[0]
