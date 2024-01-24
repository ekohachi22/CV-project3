import cv2

import onnxruntime as ort
import numpy as np

_model = ort.InferenceSession("api/model/weights/landmark.onnx")
INPUT_SIZE = (224, 224, 1)

def infer(img: np.array) -> list:
    if img.shape != INPUT_SIZE:
        img = cv2.resize(img, INPUT_SIZE[:2])
    input = np.array([img], dtype=np.float32)
    return _model.run(None, {"input": input.reshape((1,)+INPUT_SIZE)})[0]
