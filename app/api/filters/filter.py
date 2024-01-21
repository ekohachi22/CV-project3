import cv2
import json
import os

import numpy as np

from api.model.landmark_points_inference import INPUT_SIZE

class Filter:
    def __init__(self, json_descr_path: str):
        with open(json_descr_path, 'r') as f:
            data = json.load(f)
            self.image = cv2.imread(f"{os.path.split(json_descr_path)[0]}/{data['file_path']}")
            self.ref_points = data['reference_points']
    def warp_to_points(self, points: dict, size = INPUT_SIZE) -> np.array:
        dst = []
        src = []
        for keypoint_name in self.ref_points:
            if keypoint_name in points:
                dst.append(points[keypoint_name])
                src.append(self.ref_points[keypoint_name])
        assert len(dst) == len(src) == 4
        src = np.array(src, dtype=np.float32)
        dst = np.array(dst, dtype=np.float32)
        matrix = cv2.getPerspectiveTransform(src, dst)
        return cv2.warpPerspective(self.image, matrix, (size[1], size[0]), cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)