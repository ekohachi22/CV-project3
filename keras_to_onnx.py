import tf2onnx
import argparse
import tensorflow as tf
import onnxruntime as rt

from tensorflow import keras as keras

def to_onnx(input_path: str, output_path: str, image_size: int):
    model = keras.models.load_model(input_path, safe_mode=False)
    spec = (tf.TensorSpec((None, image_size, image_size, 1), tf.float32, name="input"),)
    tf2onnx.convert.from_keras(model, input_signature = spec, opset=13, output_path=output_path)

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--input_path', required=True)
    argparser.add_argument('--output_path', required=False, default=None)
    argparser.add_argument('--image_size', required=False, default=224)
    args = argparser.parse_args()
    input_path = args.input_path
    output_path = args.output_path
    image_size = int(args.image_size)
    if output_path is None:
        split = input_path.split(".")
        output_path = split[0] + ".onnx"
    to_onnx(input_path, output_path, image_size)