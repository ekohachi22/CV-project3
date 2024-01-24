import onnxruntime as rt
import numpy as np
import tensorflow as tf

providers = ['CPUExecutionProvider']
m = rt.InferenceSession("model2_Adam_mean_squared_error.onnx", providers=providers)
x = np.random.random(size = (1, 96, 96, 1)).astype(np.float32)
onnx_pred = m.run(None, {"input": x})
print(onnx_pred)