import { useEffect, useRef, useState } from "react";
import { InferenceSession, Tensor } from "onnxruntime-web";
import { env } from "onnxruntime-web/webgpu";

env.webgl.pack = true;
env.webgl.packMax = true;

const MODEL_PATH = "/models/yolo11n.onnx";
const INPUT_SIZE = 640;

function letterbox(video: HTMLVideoElement) {
    const canvas = document.createElement("canvas");
    canvas.width = INPUT_SIZE;
    canvas.height = INPUT_SIZE;
    const ctx = canvas.getContext("2d")!;

    const scale = Math.min(INPUT_SIZE / video.videoWidth, INPUT_SIZE / video.videoHeight);
    const newW = Math.round(video.videoWidth * scale);
    const newH = Math.round(video.videoHeight * scale);
    const padX = (INPUT_SIZE - newW) / 2;
    const padY = (INPUT_SIZE - newH) / 2;

    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, INPUT_SIZE, INPUT_SIZE);
    ctx.drawImage(
        video,
        0, 0, video.videoWidth, video.videoHeight,
        padX, padY, newW, newH
    );

    const img = ctx.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE).data;
    const arr = new Float32Array(3 * INPUT_SIZE * INPUT_SIZE);
    for (let i = 0; i < INPUT_SIZE * INPUT_SIZE; i++) {
        arr[i] = img[i * 4] / 255;
        arr[i + INPUT_SIZE * INPUT_SIZE] = img[i * 4 + 1] / 255;
        arr[i + 2 * INPUT_SIZE * INPUT_SIZE] = img[i * 4 + 2] / 255;
    }

    return {
        tensor: new Tensor("float32", arr, [1, 3, INPUT_SIZE, INPUT_SIZE]),
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
    const [, nBoxes, dims] = output.dims; // [1,300,6]
    const data = output.data as Float32Array;

    ctx.strokeStyle = "red";
    ctx.fillStyle = "red";
    ctx.lineWidth = 2;
    ctx.font = "14px sans-serif";

    for (let i = 0; i < nBoxes; i++) {
        const off = i * dims;
        const x1_640 = data[off + 0];
        const y1_640 = data[off + 1];
        const x2_640 = data[off + 2];
        const y2_640 = data[off + 3];
        const conf = data[off + 4];
        const clsId = Math.round(data[off + 5]);

        if (conf < 0.5) continue;

        const x1 = (x1_640 - padX) / scale;
        const y1 = (y1_640 - padY) / scale;
        const x2 = (x2_640 - padX) / scale;
        const y2 = (y2_640 - padY) / scale;
        const w = x2 - x1;
        const h = y2 - y1;

        ctx.strokeRect(x1, y1, w, h);
        const label = classNames[clsId] ?? `cls ${clsId}`;
        ctx.fillText(`${label} ${(conf * 100).toFixed(0)}%`, x1 + 2, y1 - 4);
    }
}

const YoloWebcam: React.FC = () => {
    const vidRef = useRef<HTMLVideoElement>(null);
    const canRef = useRef<HTMLCanvasElement>(null);
    const sessRef = useRef<InferenceSession | null>(null);
    const rafRef = useRef<number | undefined>(undefined);
    const [classNames, setClassNames] = useState<string[]>([]);

    useEffect(() => {
        fetch("/classes/coco80.names.json")
            .then(r => r.json())
            .then(setClassNames)
            .catch(console.error);
    }, []);

    useEffect(() => {
        if (!classNames.length) return;
        (async () => {
            console.log("WebGL supported:", !!window.WebGLRenderingContext);
            console.log("WebGPU supported:", "gpu" in navigator);

            // Try WebGL → WebGPU → WASM
            try {
                sessRef.current = await InferenceSession.create(MODEL_PATH, {
                    executionProviders: ["webgl"],
                });
                console.log("Using WebGL");
            } catch {
                try {
                    sessRef.current = await InferenceSession.create(MODEL_PATH, {
                        executionProviders: ["webgpu"],
                    });
                    console.log("Using WebGPU");
                } catch {
                    sessRef.current = await InferenceSession.create(MODEL_PATH, {
                        executionProviders: ["wasm"],
                    });
                    console.log("Using WASM");
                }
            }

            const stream = await navigator.mediaDevices.getUserMedia({ video: true });
            vidRef.current!.srcObject = stream;
            await new Promise<void>(r => {
                vidRef.current!.onloadedmetadata = () => { vidRef.current!.play(); r(); };
            });

            const loop = async () => {
                const video = vidRef.current!;
                const canvas = canRef.current!;
                const ctx = canvas.getContext("2d")!;
                if (canvas.width !== video.videoWidth) {
                    canvas.width = video.videoWidth;
                    canvas.height = video.videoHeight;
                }

                const { tensor, scale, padX, padY } = letterbox(video);
                const res = await sessRef.current!.run({
                    [sessRef.current!.inputNames[0]]: tensor
                });
                const output = res[sessRef.current!.outputNames[0]];

                ctx.drawImage(video, 0, 0);
                renderBoxes(ctx, output, scale, padX, padY, classNames);

                rafRef.current = requestAnimationFrame(loop);
            };
            loop();
        })();

        return () => {
            if (rafRef.current) cancelAnimationFrame(rafRef.current);
            if (vidRef.current?.srcObject) {
                (vidRef.current.srcObject as MediaStream).getTracks().forEach(t => t.stop());
            }
        };
    }, [classNames]);

    return (
        <div style={{
            maxWidth: "80vw",
            width: "100%",
            maxHeight: "80vh",
            height: "100%"
        }}>
            <video ref={vidRef} style={{ display: "none" }} muted playsInline />
            <canvas
                ref={canRef}
                style={{
                    width: "100%",
                    height: "100%",
                    maxHeight: "80vh",
                    objectFit: "contain",
                    borderRadius: 8,
                    display: "block",
                    margin: "0 auto"
                }}
            />
        </div>
    );
};

export default YoloWebcam;
