import cv2
import numpy as np
from rknnlite.api import RKNNLite

MODEL_PATH = "best_rknn_model/best-rk3588.rknn"
IMAGE_PATH = "test.jpg"

rknn = RKNNLite()

ret = rknn.load_rknn(MODEL_PATH)
if ret != 0:
    raise RuntimeError("Failed to load RKNN model")

ret = rknn.init_runtime()
if ret != 0:
    raise RuntimeError("Failed to init RKNN runtime")

img = cv2.imread(IMAGE_PATH)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (736, 736))
img = np.expand_dims(img, axis=0)

outputs = rknn.inference(inputs=[img])

print(type(outputs), len(outputs))
for i, out in enumerate(outputs):
    print(i, out.shape, out.dtype)

rknn.release()
