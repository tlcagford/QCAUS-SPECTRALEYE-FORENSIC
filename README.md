# 🔬 QCAUS-SpectralEye Forensic

**Professional Image & Video Spectral Analysis Platform v3.0.0**

Single-file, production-ready forensic analysis tool using Fourier-domain spectral analysis.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.28.0+-red.svg)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GPU Support](https://img.shields.io/badge/GPU-CUDA%20Enabled-green.svg)](https://developer.nvidia.com/cuda-zone)

## 🚀 Features

### 🔍 Authentication Forensics
- **Deepfake Detection** - GAN upsampling artifact identification with ML
- **JPEG Ghost Analysis** - Periodic 8×8 block signature detection
- **Copy-Move Forgery** - Phase correlation detection (placeholder)
- **Camera Sensor Fingerprinting** - Fixed-pattern noise extraction
- **Recompression Detection** - Double JPEG quantization artifacts

### 📊 Quality Assurance
- **Focus Scoring** - Automated sharpness analysis
- **Motion Blur Analysis** - Direction and magnitude estimation
- **Dead/Hot Pixel Detection** - Sensor defect mapping
- **Texture Uniformity** - Manufacturing QA analysis
- **Noise Level Assessment** - Sensor noise estimation

### 🧠 Advanced Features
- **GPU Acceleration** - CuPy-powered FFT processing (optional)
- **ML Anomaly Detection** - Isolation forest-based anomaly detection
- **Database Persistence** - SQLite/PostgreSQL support (optional)
- **Batch Processing** - Process multiple images
- **Professional Reports** - PDF generation with full analysis

### 🔐 Security
- **SSRF Protection** - Secure URL fetching
- **File Validation** - Magic bytes verification
- **Input Sanitization** - Path traversal prevention
- **Rate Limiting** - API protection (configurable)

## 📦 Quick Start

### Minimal Installation
```bash
# Install core dependencies
pip install streamlit opencv-python-headless numpy Pillow matplotlib reportlab

# Run the application
streamlit run app.py
