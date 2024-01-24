import numpy as np

from api.filters.filter import Filter
from api.model.landmark_points_inference import INPUT_SIZE

keypoint_names = [
    "left_eye_center",
    "right_eye_center",
    "left_eye_inner_corner",
    "left_eye_outer_corner",
    "right_eye_inner_corner",
    "right_eye_outer_corner",
    "left_eyebrow_inner_end",
    "left_eyebrow_outer_end",
    "right_eyebrow_inner_end",
    "right_eyebrow_outer_end",
    "nose_tip",
    "mouth_left_corner",
    "mouth_right_corner",
    "mouth_center_top_lip",
    "mouth_center_bottom_lip",
]


def model_output_to_keypoints_coordinates(output: np.array) -> dict:
    ret = {}
    for i, keypoint_name in enumerate(keypoint_names):
        keypoint_coordinates = output[0][(i * 2) : (2 * i + 2)]
        ret[keypoint_name] = keypoint_coordinates
    return ret

def apply_filter_to_img(img: np.array, filter: Filter, keypoints: dict, bounds: tuple = None) -> np.array:
    if bounds is not None:
        for k in keypoints:
            keypoints[k][0] = (keypoints[k][0] / INPUT_SIZE[1]) * bounds[2] + bounds[0]
            keypoints[k][1] = (keypoints[k][1] / INPUT_SIZE[0]) * bounds[3] + bounds[1]
    warped = filter.warp_to_points(keypoints, img.shape)
    pseudo_alpha_filter = (warped > 0).astype(np.uint8)
    inverse_pseudo_alpha_filter = np.ones(img.shape) - pseudo_alpha_filter
    ret = warped * pseudo_alpha_filter + inverse_pseudo_alpha_filter * img
    return ret