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
    faces = find_faces(img_gray)
    img_annotated = img.copy()
    face_images = []
    for x, y, w, h in faces:
        if w % 96 != 0:
            closest_multiple = np.ceil(w / 96) * 96
            diff = INPUT_SIZE[1] - closest_multiple
            x -= int(diff//2)
            if x < 0:
                w += abs(x)
                x = 0
            w += int(diff//2)
        if h % 96 != 0:
            closest_multiple = np.ceil(w / 96) * 96
            diff = INPUT_SIZE[0] - closest_multiple
            y -= int(diff//2)
            if y < 0:
                h += abs(y)
                y = 0
            h += int(diff//2)
        print(x, y, w, h)
        img_annotated = cv2.rectangle(img_annotated, (x, y), (x+w, y+h), color=(255, 0, 0), thickness=3)
        face_images.append(img_gray[y:(y+h), x:(x+w)])
        
    landmark_points = filter_utils.model_output_to_keypoints_coordinates(infer(img_gray))
    glasses = Filter('api/filters/data/sunglasses.json')
    if len(face_images) > 0:
        face_img = filter_utils.apply_filter_to_img(face_images[0], glasses, landmark_points)
        return encode_image(face_img)
    else:
        return encode_image(img_annotated)