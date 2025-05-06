import onnxruntime as ort

sess = ort.InferenceSession("yolo11n.onnx")
print("Inputs:", sess.get_inputs())
print("Outputs:", sess.get_outputs())
# Run one dummy frame through and inspect shape & first row:
import numpy as np
dummy = np.zeros((1,3,640,640), dtype=np.float32)
out = sess.run(None, {sess.get_inputs()[0].name: dummy})[0]
print("out.shape:", out.shape)
print("out[0]:", out[0])
