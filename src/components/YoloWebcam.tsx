import { useEffect, useRef, useState } from "react";
import { InferenceSession, Tensor } from "onnxruntime-web";
import { env } from "onnxruntime-web/webgpu";
import type { ModelConfig } from "./ModelSelector";
import type { CameraDevice } from "./CameraSelector";

env.webgl.pack = true;
env.webgl.packMax = true;

const DETECT_INTERVAL_MS = 40;

function letterbox(video: HTMLVideoElement, inputSize: number) {
  const canvas = Object.assign(document.createElement("canvas"), {
    width: inputSize,
    height: inputSize,
  });
  const ctx = canvas.getContext("2d")!;

  const scale = Math.min(
    inputSize / video.videoWidth,
    inputSize / video.videoHeight
  );
  const newW = Math.round(video.videoWidth * scale);
  const newH = Math.round(video.videoHeight * scale);
  const padX = (inputSize - newW) / 2;
  const padY = (inputSize - newH) / 2;

  ctx.fillStyle = "black";
  ctx.fillRect(0, 0, inputSize, inputSize);
  ctx.drawImage(
    video,
    0,
    0,
    video.videoWidth,
    video.videoHeight,
    padX,
    padY,
    newW,
    newH
  );

  const imgData = ctx.getImageData(0, 0, inputSize, inputSize).data;
  const arr = new Float32Array(3 * inputSize * inputSize);
  for (let i = 0; i < inputSize * inputSize; i++) {
    arr[i] = imgData[i * 4] / 255;
    arr[i + inputSize * inputSize] = imgData[i * 4 + 1] / 255;
    arr[i + 2 * inputSize * inputSize] = imgData[i * 4 + 2] / 255;
  }

  return {
    tensor: new Tensor("float32", arr, [1, 3, inputSize, inputSize]),
    scale,
    padX,
    padY,
  };
}

function renderBoxes(
  ctx: CanvasRenderingContext2D,
  output: Tensor,
  scale: number,
  padX: number,
  padY: number,
  classNames: string[]
) {
  const [, n, d] = output.dims;
  const data = output.data as Float32Array;
  ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
  ctx.strokeStyle = ctx.fillStyle = "red";
  ctx.lineWidth = 2;
  ctx.font = "14px sans-serif";

  for (let i = 0; i < n; i++) {
    const off = i * d;
    const conf = data[off + 4];
    if (conf < 0.5) continue;

    const [x1, y1, x2, y2] = [0, 1, 2, 3].map(
      (j) => (data[off + j] - (j % 2 ? padY : padX)) / scale
    );
    const [w, h] = [x2 - x1, y2 - y1];

    ctx.strokeRect(x1, y1, w, h);
    const clsId = Math.round(data[off + 5]);
    const label = classNames[clsId] ?? `cls ${clsId}`;
    ctx.fillText(`${label} ${(conf * 100).toFixed(0)}%`, x1 + 2, y1 - 4);
  }
}

interface YoloWebcamProps {
  modelConfig: ModelConfig;
  selectedCamera: CameraDevice | null;
}

const YoloWebcam: React.FC<YoloWebcamProps> = ({
  modelConfig,
  selectedCamera,
}) => {
  const vidRef = useRef<HTMLVideoElement>(null);
  const canRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const sessRef = useRef<InferenceSession | null>(null);
  const boxesRef = useRef<{
    out: Tensor;
    s: number;
    x: number;
    y: number;
  } | null>(null);
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const [_, setClassNames] = useState<string[]>([]);
  const [inferenceTime, setInferenceTime] = useState<number>(0);
  const streamRef = useRef<MediaStream | null>(null);

  useEffect(() => {
    let rafID: number;
    let timerID: ReturnType<typeof setTimeout>;
    let isActive = true;

    (async () => {
      try {
        const res = await fetch(modelConfig.classesPath);
        const classNames = await res.json();
        if (!isActive) return;
        setClassNames(classNames);

        const createSession = async (providers: string[]) =>
          await InferenceSession.create(modelConfig.modelPath, {
            executionProviders: providers,
          });

        sessRef.current = await createSession(["webgl"]).catch(() =>
          createSession(["webgpu"]).catch(() => createSession(["wasm"]))
        );

        if (!isActive || !selectedCamera) return;

        if (streamRef.current) {
          streamRef.current.getTracks().forEach((track) => track.stop());
        }

        const stream = await navigator.mediaDevices.getUserMedia({
          video: {
            deviceId: { exact: selectedCamera.deviceId },
            width: { ideal: 1080 },
            height: { ideal: 612 },
          },
        });

        if (!isActive) return stream.getTracks().forEach((t) => t.stop());

        streamRef.current = stream;
        vidRef.current!.srcObject = stream;

        await new Promise<void>((resolve) => {
          vidRef.current!.onloadedmetadata = () => {
            const v = vidRef.current!;
            const c = containerRef.current!;
            c.style.width = "80vw";
            c.style.maxWidth = `${v.videoWidth}px`;
            v.play();
            resolve();
          };
        });

        const draw = () => {
          const v = vidRef.current!;
          const c = canRef.current!;
          const ctx = c.getContext("2d")!;
          if (c.width !== v.videoWidth || c.height !== v.videoHeight) {
            c.width = v.videoWidth;
            c.height = v.videoHeight;
          }
          ctx.drawImage(v, 0, 0);
          const b = boxesRef.current;
          if (b) renderBoxes(ctx, b.out, b.s, b.x, b.y, classNames);
          rafID = requestAnimationFrame(draw);
        };

        const detect = async () => {
          if (!sessRef.current) return;
          const start = performance.now();
          const { tensor, scale, padX, padY } = letterbox(
            vidRef.current!,
            modelConfig.imgsz
          );

          let result;
          try {
            result = await sessRef.current.run({
              [sessRef.current.inputNames[0]]: tensor,
            });
          } catch (e) {
            console.warn("WebGPU inference failed, falling back to Wasm:", e);
            sessRef.current = await InferenceSession.create(
              modelConfig.modelPath,
              { executionProviders: ["wasm"] }
            );
            result = await sessRef.current.run({
              [sessRef.current.inputNames[0]]: tensor,
            });
          }

          if (!isActive) return;

          setInferenceTime(performance.now() - start);
          boxesRef.current = {
            out: result[sessRef.current.outputNames[0]],
            s: scale,
            x: padX,
            y: padY,
          };

          timerID = setTimeout(detect, DETECT_INTERVAL_MS);
        };

        draw();
        detect();
      } catch (err) {
        console.error("Error during setup:", err);
      }
    })();

    return () => {
      isActive = false;
      cancelAnimationFrame(rafID);
      clearTimeout(timerID);
      boxesRef.current = null;
      sessRef.current = null;
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
        streamRef.current = null;
      }
    };
  }, [modelConfig, selectedCamera]);

  return (
    <div ref={containerRef} style={{ position: "relative", margin: "0 auto" }}>
      <video
        ref={vidRef}
        muted
        playsInline
        style={{ width: "100%", height: "auto", display: "block" }}
      />
      <canvas
        ref={canRef}
        style={{
          position: "absolute",
          top: 0,
          left: 0,
          width: "100%",
          height: "auto",
          pointerEvents: "none",
        }}
      />
      <div
        style={{
          textAlign: "center",
          padding: "0.5rem",
          fontSize: "0.9rem",
          color: "#666",
        }}
      >
        Inference Time: {inferenceTime.toFixed(1)} ms
      </div>
    </div>
  );
};

export default YoloWebcam;
