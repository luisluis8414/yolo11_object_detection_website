import { useEffect, useRef } from "react";
import * as ort from "onnxruntime-web";

// Configure ONNX Runtime Web
(ort as any).env.wasm.wasmPaths = {
    'ort-wasm.wasm': '/onnxruntime-web/ort-wasm.wasm',
    'ort-wasm-simd.wasm': '/onnxruntime-web/ort-wasm-simd.wasm',
    'ort-wasm-threaded.wasm': '/onnxruntime-web/ort-wasm-threaded.wasm'
};

const MODEL_PATH = "/models/yolo11n.onnx";

function drawBoxes(ctx: CanvasRenderingContext2D, output: ort.Tensor) {
    const rows = output.dims.length === 3 ? output.dims[1] : output.dims[0];
    const cols = output.dims.length === 3 ? output.dims[2] : output.dims[1];
    const data = output.data as Float32Array;

    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);

    for (let i = 0; i < rows; i++) {
        const offset = i * cols;
        const conf = data[offset + 4];
        if (conf < 0.5) continue;

        const classScores = data.slice(offset + 5, offset + cols);
        const classIndex = classScores.indexOf(Math.max(...classScores));

        const x = data[offset + 0];
        const y = data[offset + 1];
        const w = data[offset + 2];
        const h = data[offset + 3];

        const x1 = x - w / 2;
        const y1 = y - h / 2;

        ctx.strokeStyle = "red";
        ctx.lineWidth = 2;
        ctx.strokeRect(x1, y1, w, h);

        ctx.fillStyle = "red";
        ctx.font = "16px sans-serif";
        ctx.fillText(`${conf.toFixed(2)} (class ${classIndex})`, x1, y1 - 4);
    }
}


function preprocess(imageData: ImageData): ort.Tensor {
    // Resize to 640x640
    const ctx = document.createElement("canvas").getContext("2d", { willReadFrequently: true })!;
    ctx.canvas.width = 640;
    ctx.canvas.height = 640;
    ctx.putImageData(imageData, 0, 0);
    const scaled = ctx.getImageData(0, 0, 640, 640);

    // Normalize [0,255] → [0,1], reorder to CHW
    const input = new Float32Array(1 * 3 * 640 * 640);
    for (let i = 0; i < 640 * 640; i++) {
        input[i] = scaled.data[i * 4 + 0] / 255; // R
        input[i + 640 * 640] = scaled.data[i * 4 + 1] / 255; // G
        input[i + 2 * 640 * 640] = scaled.data[i * 4 + 2] / 255; // B
    }

    return new ort.Tensor("float32", input, [1, 3, 640, 640]);
}

const YoloWebcam: React.FC = () => {
    const videoRef = useRef<HTMLVideoElement>(null);
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const sessionRef = useRef<ort.InferenceSession | null>(null);
    const animationFrameRef = useRef<number | undefined>(undefined);

    useEffect(() => {
        const setup = async () => {
            try {
                const session = await ort.InferenceSession.create(MODEL_PATH, {
                    executionProviders: ["wasm"],
                });
                sessionRef.current = session;

                const stream = await navigator.mediaDevices.getUserMedia({ video: true });
                if (videoRef.current) {
                    videoRef.current.srcObject = stream;
                    // Wait for video to be ready
                    await new Promise<void>((resolve) => {
                        if (!videoRef.current) return;
                        videoRef.current.onloadedmetadata = () => {
                            if (!videoRef.current) return;
                            videoRef.current.play();
                            resolve();
                        };
                    });
                }

                detect();
            } catch (err) {
                console.error("Setup failed:", err);
            }
        };

        const detect = async () => {
            if (!videoRef.current || !canvasRef.current || !sessionRef.current) {
                animationFrameRef.current = requestAnimationFrame(detect);
                return;
            }

            const video = videoRef.current;
            const canvas = canvasRef.current;
            const ctx = canvas.getContext("2d", { willReadFrequently: true });
            if (!ctx) return;

            // Ensure video is playing and has valid dimensions
            if (video.readyState !== video.HAVE_ENOUGH_DATA ||
                video.videoWidth === 0 ||
                video.videoHeight === 0) {
                animationFrameRef.current = requestAnimationFrame(detect);
                return;
            }

            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

            const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
            const inputTensor = preprocess(imageData);

            const feeds: Record<string, ort.Tensor> = {};
            feeds[sessionRef.current.inputNames[0]] = inputTensor;

            const results = await sessionRef.current.run(feeds);
            const output = results[sessionRef.current.outputNames[0]];

            drawBoxes(ctx, output);
            animationFrameRef.current = requestAnimationFrame(detect);
        };

        setup();

        return () => {
            if (animationFrameRef.current) {
                cancelAnimationFrame(animationFrameRef.current);
            }
            if (videoRef.current?.srcObject) {
                const stream = videoRef.current.srcObject as MediaStream;
                stream.getTracks().forEach(track => track.stop());
            }
        };
    }, []);

    return (
        <div style={{
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            width: '100%',
            maxWidth: '960px',
            margin: '0 auto'
        }}>
            <video
                ref={videoRef}
                autoPlay
                playsInline
                muted
                style={{
                    width: '100%',
                    height: 'auto',
                    display: 'none'
                }}
            />
            <canvas
                ref={canvasRef}
                style={{
                    width: '100%',
                    height: 'auto',
                    backgroundColor: '#000',
                    borderRadius: '8px'
                }}
            />
        </div>
    );
};

export default YoloWebcam;
