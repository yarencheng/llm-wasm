# Gemma 3 Web LLM Demo

A high-performance, on-device Large Language Model demo running directly in your browser using MediaPipe and WebGPU.

## 🚀 Features
- **100% On-Device**: No data leaves your machine. Private and secure.
- **Hardware Accelerated**: Uses WebGPU for blazing fast inference.
- **Gemma 3 Powered**: Utilizes the latest Gemma 3 E2B model from Google.
- **Streaming UI**: Modern glassmorphism interface with real-time token generation.

## 📋 Prerequisites
- **Node.js**: v18 or higher recommended.
- **Browser**: A WebGPU-compatible browser (e.g., Chrome v113+, Edge v113+).
- **Hardware**: Dedicated or powerful integrated GPU is recommended for the 2.8GB model.

## 🛠️ Installation

1. **Clone the repository** (if applicable) or navigate to the project directory:
   ```bash
   cd llm-wasm
   ```

2. **Install dependencies**:
   ```bash
   npm install
   ```

## 🏃 How to Run

1. **Start the development server**:
   ```bash
   npm run dev
   ```

2. **Open the application**:
   Navigate to `http://localhost:5173/` in your browser.

3. **Loading the Model**:
   - The app will automatically attempt to load the `gemma-3n-E2B-it-int4-Web.litertlm` model from the `public/` directory.
   - **Note**: The model file is large (~2.8GB). The first load may take some time depending on your disk speed and browser memory allocation.

## 🧠 Model Information
- **Name**: `gemma-3n-E2B-it-int4-Web`
- **Format**: `.litertlm` (Optimized for MediaPipe Web)
- **Parameters**: 2B (quantized to 4-bit)

## 📄 License
This project is for demonstration purposes. Gemma 3 is subject to the [Gemma Terms of Use](https://ai.google.dev/gemma/terms).
