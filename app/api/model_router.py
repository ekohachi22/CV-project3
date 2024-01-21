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
        x_center = (x + w // 2) 
        y_center = (y + h // 2)
        w, h = 96 * 2, 96 * 2
        x = x_center - w//2
        y = y_center - h//2
        if x < 0:
            x = 0
        if y < 0:
            y = 0
        if x + w >= img.shape[1]:
            x += x + w - img.shape[1] - 1
        if y + h >= img.shape[0]:
            y += y + h - img.shape[0] - 1
        print(x, y, w, h)
        face_img = img_gray[y:(y+h), x:(x+w)]
        landmark_points = filter_utils.model_output_to_keypoints_coordinates(infer(face_img))
        ret = filter_utils.apply_filter_to_img(ret, glasses, landmark_points, (x, y, w, h))
    return encode_image(ret)
        