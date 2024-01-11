import pandas as pd
import cv2
import numpy as np
import os
import argparse

def _extract_images(image_data: pd.DataFrame, save_folder: str, scale_factor: float):
    for i in range(image_data.shape[0]):
        img_data = image_data.iloc[i]
        img_data_array = np.array([int(i) for i in img_data.split(" ")], dtype=np.uint8).reshape(96, 96)
        img_data_array = cv2.resize(img_data_array, (int(img_data_array.shape[0] * scale_factor), int(img_data_array.shape[1] * scale_factor)))
        cv2.imwrite(save_folder + f"/{i}.jpg", img_data_array)

def _save_annotations(df: pd.DataFrame, save_path: str):
    df_no_image = df.drop('Image', axis = 1)
    df_no_image /= 96 #Normalize to 0-1 range
    file_paths = [f"{i}.jpg" for i in range(df_no_image.shape[0])]
    df_no_image['file_path'] = file_paths
    df_no_image.to_csv(save_path + "/annotations.csv")

def _extract_images_and_annotations(csv_path: str, save_path: str, scale_factor: float):
    if os.path.exists(save_path):
        raise Exception("Output path already exists!")
    os.mkdir(save_path)
    df = pd.read_csv(csv_path)
    image_data = df.Image
    _save_annotations(df, save_path)
    _extract_images(image_data, save_path, scale_factor)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--csv_path', required=True)
    argparser.add_argument('--save_path', required=True)
    argparser.add_argument('--scale', required=False, default=3)
    args = argparser.parse_args()
    _extract_images_and_annotations(args.csv_path, args.save_path, float(args.scale))