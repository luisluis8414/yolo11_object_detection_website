from ultralytics import YOLO

# Load your pretrained model (yolo11n.pt is already trained)
model = YOLO("best.pt")

# Export to ONNX **with** built-in NMS + simplified graph + dynamic axes
model.export(
    format="onnx",
    opset=12,        # ONNX opset version
    simplify=True,   # run onnx-simplifier
    nms=True,        # bake in the grid-decode + sigmoid + NMS
    dynamic=True     # allow dynamic batch / image sizes
)
