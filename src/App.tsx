import { useState } from "react";
import YoloWebcam from "./components/YoloWebcam";
import ModelSelector, { type ModelConfig } from "./components/ModelSelector";
import CameraSelector, { type CameraDevice } from "./components/CameraSelector";

const AVAILABLE_MODELS: ModelConfig[] = [
  {
    name: "Fruit Detection Model N",
    modelPath: "/models/fruits/fruits.n.onnx",
    classesPath: "/classes/fruits.json",
    imgsz: 320,
  },
  {
    name: "Fruit Detection Model S",
    modelPath: "/models/fruits/fruits.s.onnx",
    classesPath: "/classes/fruits.json",
    imgsz: 640,
  },
  {
    name: "COCO Detection n",
    modelPath: "/models/yolo11n.onnx",
    classesPath: "/classes/coco80.names.json",
    imgsz: 640,
  },
  {
    name: "COCO Detection m",
    modelPath: "/models/yolo11m.onnx",
    classesPath: "/classes/coco80.names.json",
    imgsz: 640,
  },
];

function App() {
  const [selectedModel, setSelectedModel] = useState<ModelConfig>(
    AVAILABLE_MODELS[2]
  );
  const [selectedCamera, setSelectedCamera] = useState<CameraDevice | null>(
    null
  );

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        minHeight: "100vh",
        width: "100vw",
        maxWidth: "100%",
        padding: "2rem",
        boxSizing: "border-box",
      }}
    >
      <h1 style={{ textAlign: "center" }}>YOLOv11n ONNX Inference</h1>
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
        }}
      >
        <ModelSelector
          models={AVAILABLE_MODELS}
          selectedModel={selectedModel}
          onModelSelect={setSelectedModel}
        />
        <CameraSelector
          selectedDevice={selectedCamera}
          onDeviceSelect={setSelectedCamera}
        />
      </div>
      <YoloWebcam modelConfig={selectedModel} selectedCamera={selectedCamera} />
    </div>
  );
}

export default App;
