# Real-Time Object Detection in the Browser using YOLOv11n (ONNX)

🚀 A web application for real-time object detection using ONNX Runtime Web and React. This project allows users to perform object detection directly in their web browser using a pre-trained model.

![preview](https://object-detection-deep-learning-project.vercel.app/preview.png)

## 🧠 What It Does

This project runs a pre-trained YOLOv11n model (ONNX format) in the browser using your webcam. It performs object detection entirely client-side — no server, no uploads, no latency.

- 🔍 **Real-time detection** with bounding boxes
- 🖥️ Runs **entirely in the browser**
- 📷 Uses your **live webcam feed**

## 🌐 Live Demo

👉 [Try it here (Vercel)](https://object-detection-deep-learning-project.vercel.app)

## 🔒 Privacy

All inference runs entirely locally. No video or data is ever uploaded.

## Prerequisites

Before you begin, ensure you have the following installed:

- Node.js (Latest LTS version recommended)
- npm (comes with Node.js)

## Installation

1. Clone the repository:

```bash
git clone https://github.com/luisluis8414/object_detection_website.git
cd object_detection_website
```

2. Install dependencies:

```bash
npm install
```

## Development

To start the development server:

```bash
npm run dev
```

This will:

- Copy necessary WASM files
- Start the Vite development server
- Open the application in your default browser at `http://localhost:5173`

## Technologies Used

- React 19.1.0
- TypeScript 5.8.3
- ONNX Runtime Web 1.17.0
- Vite 6.3.5
