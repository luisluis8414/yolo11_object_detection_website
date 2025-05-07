import { useEffect, useRef, useState } from "react";
import { InferenceSession, Tensor } from "onnxruntime-web";
import { env } from "onnxruntime-web/webgpu";
import type { ModelConfig } from "./ModelSelector";

env.webgl.pack = true;
env.webgl.packMax = true;

const DETECT_INTERVAL_MS = 40;

function letterbox(video: HTMLVideoElement, inputSize: number) {
  const canvas = document.createElement("canvas");
  canvas.width = inputSize;
  canvas.height = inputSize;
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
  const [, nBoxes, dims] = output.dims;
  const data = output.data as Float32Array;
  ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
  ctx.strokeStyle = "red";
  ctx.fillStyle = "red";
  ctx.lineWidth = 2;
  ctx.font = "14px sans-serif";
  for (let i = 0; i < nBoxes; i++) {
    const off = i * dims;
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
  const sessRef = useRef<InferenceSession | null>(null);
  const boxesRef = useRef<{
    out: Tensor;
    s: number;
    x: number;
    y: number;
  } | null>(null);
  const [classNames, setClassNames] = useState<string[]>([]);

  useEffect(() => {
    fetch(modelConfig.classesPath)
      .then((r) => r.json())
      .then(setClassNames)
      .catch(console.error);
  }, [modelConfig.classesPath]);

  useEffect(() => {
    if (!classNames.length) return;
    let drawRAF: number;
    let detectTimer: ReturnType<typeof setTimeout>;

    (async () => {
      try {
        sessRef.current = await InferenceSession.create(modelConfig.modelPath, {
          executionProviders: ["webgl"],
        });
      } catch {
        try {
          sessRef.current = await InferenceSession.create(
            modelConfig.modelPath,
            {
              executionProviders: ["webgpu"],
            }
          );
        } catch {
          sessRef.current = await InferenceSession.create(
            modelConfig.modelPath,
            {
              executionProviders: ["wasm"],
            }
          );
        }
      }
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      vidRef.current!.srcObject = stream;
      await new Promise<void>((r) => {
        vidRef.current!.onloadedmetadata = () => {
          vidRef.current!.play();
          r();
        };
      });

      const draw = () => {
        const video = vidRef.current!;
        const canvas = canRef.current!;
        const ctx = canvas.getContext("2d")!;
        if (canvas.width !== video.videoWidth) {
          canvas.width = video.videoWidth;
          canvas.height = video.videoHeight;
        }
        ctx.drawImage(video, 0, 0);
        const b = boxesRef.current;
        if (b) renderBoxes(ctx, b.out, b.s, b.x, b.y, classNames);
        drawRAF = requestAnimationFrame(draw);
      };

      const detect = async () => {
        const { tensor, scale, padX, padY } = letterbox(vidRef.current!, modelConfig.imgsz);
        const res = await sessRef.current!.run({
          [sessRef.current!.inputNames[0]]: tensor,
        });
        boxesRef.current = {
          out: res[sessRef.current!.outputNames[0]],
          s: scale,
          x: padX,
          y: padY,
        };
        detectTimer = setTimeout(detect, DETECT_INTERVAL_MS);
      };

      drawRAF = requestAnimationFrame(draw);
      detect();
    })();

    return () => {
      cancelAnimationFrame(drawRAF);
      clearTimeout(detectTimer);
      if (vidRef.current?.srcObject) {
        (vidRef.current.srcObject as MediaStream)
          .getTracks()
          .forEach((t) => t.stop());
      }
      if (sessRef.current) {
        sessRef.current = null;
      }
    };
  }, [classNames, modelConfig.modelPath, modelConfig.imgsz]);

  return (
    <div
      style={{
        position: "relative",
        width: "80vw",
        maxHeight: "80vh",
        margin: "0 auto",
      }}
    >
      <video
        ref={vidRef}
        muted
        playsInline
        style={{
          width: "100%",
          height: "auto",
          maxHeight: "80vh",
          objectFit: "contain",
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
          maxHeight: "80vh",
          pointerEvents: "none",
        }}
      />
    </div>
  );
};

export default YoloWebcam;
