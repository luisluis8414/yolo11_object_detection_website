import YoloWebcam from "./components/YoloWebcam";

function App() {
  return (
    <div style={{
      display: "flex",
      flexDirection: "column",
      alignItems: "center",
      minHeight: "100vh",
      width: "100vw",
      maxWidth: "100%",
      padding: "2rem",
      boxSizing: "border-box"
    }}>
      <h1 style={{ textAlign: "center", marginBottom: "2rem" }}>YOLOv11n ONNX Inference</h1>
      <YoloWebcam />
    </div>
  );
}

export default App;
