import base64
import cv2
import numpy as np

from fastapi import APIRouter

from api.pydantic_models.base64_encoded_model import Base64InputEncodedModel, Base64OutputEncodedModel
from api.model.haar_cascade import find_faces
from api.model.landmark_points_inference import infer, INPUT_SIZE
from api.filters import filter_utils
from api.filters.filter import Filter

model_router = APIRouter()

def decode_image(data: Base64InputEncodedModel) -> np.array:
    img = np.frombuffer(data.img_bytes, dtype=np.uint8)
    return cv2.imdecode(img, cv2.IMREAD_COLOR)

def encode_image(arr: np.array) -> Base64OutputEncodedModel:
    _, buf = cv2.imencode('.png', arr)
    base64_str = base64.b64encode(buf)
    return Base64OutputEncodedModel(img_str = base64_str)

@model_router.post("/transform")
async def transform(data: Base64InputEncodedModel) -> Base64OutputEncodedModel:
    img = decode_image(data)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = find_faces(img)
    glasses = Filter('api/filters/data/sunglasses.json')
    ret = img.copy()
    for x, y, w, h in faces:
        print(x, y, w, h)
        face_img = img_gray[y:(y+h), x:(x+w)]
        inf = infer(face_img)
        print(inf)
        landmark_points = filter_utils.model_output_to_keypoints_coordinates(inf/100)
        ret = filter_utils.apply_filter_to_img(ret, glasses, landmark_points, (x, y))
    return encode_image(ret)
        