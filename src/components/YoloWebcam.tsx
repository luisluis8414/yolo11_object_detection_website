import { useEffect, useRef, useState } from "react";
import { InferenceSession, Tensor } from "onnxruntime-web";
import { env } from "onnxruntime-web/webgpu";
import type { ModelConfig } from "./ModelSelector";

env.webgl.pack = true;
env.webgl.packMax = true;

const DETECT_INTERVAL_MS = 40;

function letterbox(video: HTMLVideoElement, inputSize: number) {
  const offscreen = document.createElement("canvas");
  offscreen.width = inputSize;
  offscreen.height = inputSize;
  const ctx = offscreen.getContext("2d")!;
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
  const img = ctx.getImageData(0, 0, inputSize, inputSize).data;
  const arr = new Float32Array(3 * inputSize * inputSize);
  for (let i = 0; i < inputSize * inputSize; i++) {
    arr[i] = img[i * 4] / 255;
    arr[i + inputSize * inputSize] = img[i * 4 + 1] / 255;
    arr[i + 2 * inputSize * inputSize] = img[i * 4 + 2] / 255;
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
  ctx.strokeStyle = "red";
  ctx.fillStyle = "red";
  ctx.lineWidth = 2;
  ctx.font = "14px sans-serif";

  for (let i = 0; i < n; i++) {
    const off = i * d;
    const conf = data[off + 4];
    if (conf < 0.5) continue;

    const x1 = (data[off] - padX) / scale;
    const y1 = (data[off + 1] - padY) / scale;
    const x2 = (data[off + 2] - padX) / scale;
    const y2 = (data[off + 3] - padY) / scale;
    const w = x2 - x1;
    const h = y2 - y1;

    ctx.strokeRect(x1, y1, w, h);
    const clsId = Math.round(data[off + 5]);
    const label = classNames[clsId] ?? `cls ${clsId}`;
    ctx.fillText(`${label} ${(conf * 100).toFixed(0)}%`, x1 + 2, y1 - 4);
  }
}

interface YoloWebcamProps {
  modelConfig: ModelConfig;
}

const YoloWebcam: React.FC<YoloWebcamProps> = ({ modelConfig }) => {
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
  const [classNames, setClassNames] = useState<string[]>([]);
  const [inferenceTime, setInferenceTime] = useState<number>(0);
  const [videoDevices, setVideoDevices] = useState<MediaDeviceInfo[]>([]);
  const [currentDeviceId, setCurrentDeviceId] = useState<string>("");
  const streamRef = useRef<MediaStream | null>(null);

  useEffect(() => {
    async function getVideoDevices() {
      try {
        await navigator.mediaDevices.getUserMedia({ video: true })
          .then(stream => {
            stream.getTracks().forEach(track => track.stop());
          });

        const devices = await navigator.mediaDevices.enumerateDevices();
        const videoDevs = devices.filter(device => device.kind === 'videoinput');
        setVideoDevices(videoDevs);
        if (videoDevs.length > 0 && !currentDeviceId) {
          setCurrentDeviceId(videoDevs[0].deviceId);
        }
      } catch (error) {
        console.error('Error getting video devices:', error);
      }
    }

    navigator.mediaDevices.addEventListener('devicechange', getVideoDevices);
    getVideoDevices();

    return () => {
      navigator.mediaDevices.removeEventListener('devicechange', getVideoDevices);
    };
  }, []);

  useEffect(() => {
    fetch(modelConfig.classesPath)
      .then((r) => r.json())
      .then(setClassNames)
      .catch(console.error);
  }, [modelConfig.classesPath]);

  useEffect(() => {
    if (!classNames.length || !currentDeviceId) return;
    let rafID: number;
    let timerID: ReturnType<typeof setTimeout>;

    (async () => {
      try {
        sessRef.current = await InferenceSession.create(modelConfig.modelPath, {
          executionProviders: ["webgl"],
        });
      } catch {
        try {
          sessRef.current = await InferenceSession.create(
            modelConfig.modelPath,
            { executionProviders: ["webgpu"] }
          );
        } catch {
          sessRef.current = await InferenceSession.create(
            modelConfig.modelPath,
            { executionProviders: ["wasm"] }
          );
        }
      }

      // Stop any existing stream
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
      }

      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          deviceId: { exact: currentDeviceId },
          width: { ideal: 1000 },
          height: { ideal: 580 }
        },
      });
      streamRef.current = stream;
      vidRef.current!.srcObject = stream;

      await new Promise<void>((r) => {
        vidRef.current!.onloadedmetadata = () => {
          const v = vidRef.current!;
          const c = containerRef.current!;
          c.style.width = "80vw";
          c.style.maxWidth = `${v.videoWidth}px`;
          v.play();
          r();
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
        const startTime = performance.now();
        const { tensor, scale, padX, padY } = letterbox(
          vidRef.current!,
          modelConfig.imgsz
        );
        let res;
        try {
          res = await sessRef.current!.run({ [sessRef.current!.inputNames[0]]: tensor });
        } catch (e) {
          console.warn("WebGPU inference failed, falling back to Wasm:", e);
          sessRef.current = await InferenceSession.create(modelConfig.modelPath, {
            executionProviders: ["wasm"],
          });
          res = await sessRef.current!.run({ [sessRef.current!.inputNames[0]]: tensor });
        }
        const endTime = performance.now();
        setInferenceTime(endTime - startTime);
        boxesRef.current = {
          out: res[sessRef.current!.outputNames[0]],
          s: scale,
          x: padX,
          y: padY,
        };
        timerID = setTimeout(detect, DETECT_INTERVAL_MS);
      };

      draw();
      detect();
    })();

    return () => {
      cancelAnimationFrame(rafID);
      clearTimeout(timerID);
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((t) => t.stop());
      }
      sessRef.current = null;
    };
  }, [classNames, modelConfig.modelPath, modelConfig.imgsz, currentDeviceId]);

  return (
    <div
      ref={containerRef}
      style={{
        position: "relative",
        margin: "0 auto",
      }}
    >
      <div style={{
        display: 'flex',
        justifyContent: 'center',
        marginBottom: '1rem',
      }}>
        {videoDevices.length > 1 && (
          <div style={{ display: 'flex', alignItems: 'center' }}>
            <label style={{ marginRight: '0.5rem' }}>Camera: </label>
            <select
              value={currentDeviceId}
              onChange={(e) => setCurrentDeviceId(e.target.value)}
              style={{
                padding: '0.5rem',
                borderRadius: '4px',
                border: '1px solid #ccc',
                fontSize: '1rem',
              }}
            >
              {videoDevices.map((device) => (
                <option key={device.deviceId} value={device.deviceId}>
                  {device.label || `Camera ${videoDevices.indexOf(device) + 1}`}
                </option>
              ))}
            </select>
          </div>
        )}
      </div>
      <video
        ref={vidRef}
        muted
        playsInline
        style={{
          width: "100%",
          height: "auto",
          display: "block",
        }}
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
      <div style={{
        textAlign: "center",
        padding: "0.5rem",
        fontSize: "0.9rem",
        color: "#666",
      }}>
        Inference Time: {inferenceTime.toFixed(1)} ms
      </div>
    </div>
  );
};

export default YoloWebcam;
