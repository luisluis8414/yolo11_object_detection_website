import { useEffect, useRef } from "react";
import * as ort from "onnxruntime-web";

/* ── WASM runtime paths ─────────────────────────────────────────────────── */
; (ort as any).env.wasm.wasmPaths = {
    "ort-wasm.wasm": "/onnxruntime-web/ort-wasm.wasm",
    "ort-wasm-simd.wasm": "/onnxruntime-web/ort-wasm-simd.wasm",
    "ort-wasm-threaded.wasm": "/onnxruntime-web/ort-wasm-threaded.wasm",
};

const MODEL_PATH = "/models/yolo11n.onnx";
const INPUT_SIZE = 640;

/* ── letter-box + build tensor ────────────────────────────────────────── */
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

    const data = ctx.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE).data;
    const arr = new Float32Array(3 * INPUT_SIZE * INPUT_SIZE);
    for (let i = 0; i < INPUT_SIZE * INPUT_SIZE; i++) {
        arr[i] = data[i * 4] / 255;
        arr[i + INPUT_SIZE * INPUT_SIZE] = data[i * 4 + 1] / 255;
        arr[i + 2 * INPUT_SIZE * INPUT_SIZE] = data[i * 4 + 2] / 255;
    }

    return {
        tensor: new ort.Tensor("float32", arr, [1, 3, INPUT_SIZE, INPUT_SIZE]),
        scale,
        padX,
        padY,
    };
}

function renderBoxes(
    ctx: CanvasRenderingContext2D,
    output: ort.Tensor,
    scale: number,
    padX: number,
    padY: number
) {
    const [, nBoxes, dims] = output.dims;
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

        if (conf < 0.5) continue;

        const x1 = (x1_640 - padX) / scale;
        const y1 = (y1_640 - padY) / scale;
        const x2 = (x2_640 - padX) / scale;
        const y2 = (y2_640 - padY) / scale;

        const w = x2 - x1;
        const h = y2 - y1;
        ctx.strokeRect(x1, y1, w, h);
        ctx.fillText(conf.toFixed(2), x1 + 2, y1 - 4);
    }
}


/* ── React component ───────────────────────────────────────────────────── */
const YoloWebcam: React.FC = () => {
    const vidRef = useRef<HTMLVideoElement>(null);
    const canRef = useRef<HTMLCanvasElement>(null);
    const sessRef = useRef<ort.InferenceSession | null>(null);
    const rafRef = useRef<number | undefined>(undefined);

    useEffect(() => {
        (async () => {
            sessRef.current = await ort.InferenceSession.create(MODEL_PATH, {
                executionProviders: ["wasm"],
            });

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
                const feeds: Record<string, ort.Tensor> = {};
                feeds[sessRef.current!.inputNames[0]] = tensor;
                const res = await sessRef.current!.run(feeds);
                const output = res[sessRef.current!.outputNames[0]];

                ctx.drawImage(video, 0, 0);

                renderBoxes(ctx, output, scale, padX, padY);

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
    }, []);

    return (
        <div style={{ maxWidth: 960, margin: "0 auto" }}>
            <video ref={vidRef} style={{ display: "none" }} muted playsInline />
            <canvas
                ref={canRef}
                style={{ width: "100%", height: "auto", borderRadius: 8, background: "#000" }}
            />
        </div>
    );
};

export default YoloWebcam;
