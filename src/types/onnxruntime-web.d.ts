declare module 'onnxruntime-web' {
    export class InferenceSession {
        static create(path: string, options?: { executionProviders?: string[] }): Promise<InferenceSession>;
        inputNames: string[];
        outputNames: string[];
        run(feeds: Record<string, Tensor>): Promise<Record<string, Tensor>>;
    }

    export class Tensor {
        data: TypedArray;
        dims: number[];
        constructor(type: string, data: ArrayBuffer | TypedArray, dims: number[]);
    }

    type TypedArray = Int8Array | Uint8Array | Int16Array | Uint16Array | Int32Array | Uint32Array | Float32Array | Float64Array;
} 