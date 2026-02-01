# 🎯 VisionUX AI - Design Attention Predictor

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/FastAPI-0.100+-green?logo=fastapi" alt="FastAPI">
  <img src="https://img.shields.io/badge/React-18+-61DAFB?logo=react" alt="React">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-red?logo=pytorch" alt="PyTorch">
  <img src="https://img.shields.io/badge/TailwindCSS-4.0-38B2AC?logo=tailwindcss" alt="Tailwind">
</p>

**VisionUX AI** is an AI-powered visual attention prediction system that analyzes UI/UX designs and generates heatmaps showing where users are most likely to look. Built with deep learning and validated against real human eye-tracking data.

---

## ✨ Features

- 🔥 **AI-Powered Heatmap Generation** - Predicts visual attention using VGG16 feature extraction
- 📊 **Attention Score** - Quantifies how effectively a design captures user attention
- 🎨 **Interactive Comparison Slider** - Compare original design with AI-generated heatmap
- 📱 **Responsive Dashboard** - Modern dark-themed SaaS UI built with React & Tailwind
- 🔬 **Scientific Validation** - Validated against UEyes human eye-tracking benchmark
- ⚡ **Real-time Processing** - Fast inference with PyTorch backend

---

## 🖼️ Screenshots

| Dashboard     | Heatmap Analysis          |
| ------------- | ------------------------- |
| _Upload Zone_ | _Side-by-side Comparison_ |

---

## 🏗️ Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   React + Vite  │────▶│    FastAPI      │────▶│  VGG16 Model    │
│   Frontend      │◀────│    Backend      │◀────│  (PyTorch)      │
└─────────────────┘     └─────────────────┘     └─────────────────┘
     Port 5173              Port 8000            Feature Extraction
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Node.js 18+
- npm or yarn

### 1. Clone the Repository

```bash
git clone https://github.com/lianabaratian/Design-Detector.git
cd Design-Detector
```

### 2. Setup Backend

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start FastAPI server
uvicorn main:app --reload --port 8000
```

### 3. Setup Frontend

```bash
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

### 4. Open the App

Navigate to **http://localhost:5173** in your browser.

---

## 📡 API Reference

### `POST /predict`

Analyze an image and generate attention heatmap.

**Request:**

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@your-design.png"
```

**Response:**

```json
{
  "heatmap": "<base64-encoded-image>",
  "attention_score": 2.45
}
```

### `GET /`

Health check endpoint.

**Response:**

```json
{
  "message": "Saliency Prediction API. POST an image to /predict."
}
```

---

## 🔬 Model Validation

The model has been validated against the **UEyes Dataset** - a large-scale human eye-tracking benchmark with 1,980 UI screenshots.

### Validation Metrics

| Metric  | Description                  | Score            |
| ------- | ---------------------------- | ---------------- |
| **CC**  | Correlation Coefficient      | Higher is better |
| **SIM** | Similarity Score             | Higher is better |
| **NSS** | Normalized Scanpath Saliency | Higher is better |
| **AUC** | Area Under ROC Curve         | Higher is better |

### Run Validation

```bash
# With demo data
python validation/validate_model.py --demo

# With UEyes dataset
python validation/validate_model.py --dataset /path/to/UEyes_dataset
```

See [validation/README.md](validation/README.md) for detailed documentation.

---

## 📁 Project Structure

```
Design-Detector/
├── main.py                 # FastAPI backend with VGG16 model
├── requirements.txt        # Python dependencies
├── README.md              # This file
│
├── frontend/              # React dashboard
│   ├── src/
│   │   ├── App.jsx        # Main dashboard component
│   │   └── index.css      # Tailwind styles
│   ├── package.json
│   └── vite.config.js
│
├── validation/            # Model validation module
│   ├── validate_model.py  # Validation pipeline
│   ├── README.md          # Validation documentation
│   └── results/           # Generated reports
│
└── data/                  # Test samples
    └── test_samples/
```

---

## 🛠️ Tech Stack

### Backend

- **FastAPI** - Modern Python web framework
- **PyTorch** - Deep learning framework
- **TorchVision** - Pre-trained VGG16 model
- **OpenCV** - Image processing
- **NumPy** - Numerical computing

### Frontend

- **React 18** - UI library
- **Vite** - Build tool
- **Tailwind CSS 4** - Utility-first CSS
- **Axios** - HTTP client
- **react-compare-slider** - Image comparison
- **Lucide React** - Icon library

### Validation

- **SciPy** - Scientific computing
- **Matplotlib** - Visualization
- **tqdm** - Progress bars

---

## 📊 How It Works

1. **Image Upload** - User uploads a UI screenshot via the dashboard
2. **Preprocessing** - Image is resized to 224×224 while maintaining aspect ratio
3. **Feature Extraction** - VGG16 convolutional layers extract visual features
4. **Saliency Generation** - Feature maps are aggregated into attention heatmap
5. **Overlay & Score** - Heatmap is blended with original and attention score calculated
6. **Display** - Results shown in interactive comparison slider

---

## 🎯 Use Cases

- **UX Designers** - Validate designs before user testing
- **Marketing Teams** - Optimize ad creatives for attention
- **Product Managers** - Make data-driven design decisions
- **A/B Testing** - Compare attention across design variants

---

## 📚 Research References

- **UEyes Dataset**: Jiang et al., "Understanding Visual Saliency across User Interface Types" (CHI 2023)
- **VGG16**: Simonyan & Zisserman, "Very Deep Convolutional Networks for Large-Scale Image Recognition"
- **MIT Saliency Benchmark**: Bylinskii et al., "MIT/Tuebingen Saliency Benchmark"

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👩‍💻 Author

**Liana Baratian**

- GitHub: [@lianabaratian](https://github.com/lianabaratian)

---

## 🙏 Acknowledgments

- UEyes Dataset by Aalto University & University of Luxembourg
- Pre-trained VGG16 weights from PyTorch/TorchVision
- Inspired by saliency research from MIT and Google

---

<p align="center">
  Made with ❤️ for better UX
</p>
