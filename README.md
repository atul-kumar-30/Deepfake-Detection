# 🔍 DeepFake Detection System

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13%2B-orange?style=flat-square&logo=tensorflow)
![Flask](https://img.shields.io/badge/Flask-API-black?style=flat-square&logo=flask)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green?style=flat-square&logo=opencv)
![License](https://img.shields.io/badge/License-Educational-lightgrey?style=flat-square)

A full-stack **DeepFake Detection** web application built using **Flask, TensorFlow, MTCNN, and OpenCV**.

This system analyzes uploaded **images and videos** and predicts whether the content is **REAL or FAKE** using deep learning-based facial analysis powered by **EfficientNet-B4**.

---

## 🚀 Features

- 🧠 **Deep Learning Backend** — Flask REST API with TensorFlow model inference
- 🖼️ **Image Analysis** — Supports JPG and PNG formats
- 🎥 **Video Analysis** — Frame-by-frame analysis with temporal consistency checks
- 🔍 **Face Detection** — Powered by MTCNN for accurate facial localization
- 📊 **Confidence Scores** — Real/Fake probability breakdown with detailed indicators
- 🌙 **Dark / Light Mode** — Toggle-able UI theme
- 📈 **Analytics Dashboard** — Track detection statistics across sessions
- ⚡ **Demo Mode** — Works even without a trained model (simulated predictions)

---

## 🏗️ Tech Stack

### 🔹 Backend

| Package | Purpose |
|---|---|
| Flask | REST API server |
| Flask-CORS | Cross-origin request handling |
| TensorFlow / Keras | Deep learning inference |
| OpenCV | Image & video processing |
| MTCNN | Face detection |
| NumPy | Numerical operations |

### 🔹 Frontend

| Technology | Purpose |
|---|---|
| HTML5 | Structure |
| CSS3 + TailwindCSS | Styling & animations |
| JavaScript (Fetch API) | API communication & UI updates |

---

## 📂 Project Structure

```
Deepfake-Detection/
│
├── backend.py              # Flask API server & ML inference logic
├── frontend.html           # Frontend web interface (v2.0 Pro)
├── requirements.txt        # Python dependencies
│
├── notebooks/
│   └── training_complete.ipynb   # Model training notebook (DFDC + FF++ + Celeb-DF)
│
├── models/                 # Saved deep learning models
│   ├── deepfake_best_3000samples.h5    # Best validation checkpoint (Recommended)
│   └── deepfake_3000samples_final.h5   # Final epoch checkpoint
│
├── results/                # Evaluation plots
│   ├── confusion_matrix_3000.png       # Test confusion matrix plot
│   └── training_history_3000.png       # Training curves
│
├── uploads/                # Temporary file upload storage
│   └── .gitkeep
│
└── README.md
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone Repository

```bash
git clone https://github.com/atul-kumar-30/Deepfake-Detection.git
cd Deepfake-Detection
```

### 2️⃣ Create a Virtual Environment (Recommended)

```bash
conda create -n deepfake python=3.10 -y
conda activate deepfake
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

> Includes: Flask, TensorFlow, OpenCV, MTCNN, scikit-learn, matplotlib, seaborn, tqdm

### 4️⃣ Run the Backend Server

```bash
python backend.py
```

### 5️⃣ Open the Application

Open your browser and go to:

```
http://localhost:5000
```

> The frontend UI is served directly by Flask — no need to open `frontend.html` manually.

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/analyze` | Upload and analyze an image or video |
| `GET` | `/api/health` | Check backend status and model load state |
| `POST` | `/api/thumbnail` | Extract a mid-point frame from any video (base64 JPEG) |
| `POST` | `/api/convert-video` | Transcode any video to H.264 MP4 for browser playback |
| `GET` | `/api/serve-video/<id>` | Stream a previously converted video |

### Example — Analyze an image via curl:

```bash
curl -X POST -F "file=@image.jpg" http://localhost:5000/api/analyze
```

### Example API Response:

```json
{
  "success": true,
  "is_fake": false,
  "confidence": 87.4,
  "fake_probability": 12.6,
  "authentic_probability": 87.4,
  "raw_score": 0.063,
  "demo_mode": false
}
```


---

## 🧠 Model Details

| Parameter | Value |
|---|---|
| Architecture | EfficientNet-B4 |
| Face Detection | MTCNN |
| Input Size | 224 × 224 |
| Output | Binary (Real / Fake) |
| Activation | Sigmoid |
| Training Subset | 4,500 balanced samples (T4 GPU Optimized) |

### 📊 Model Performance & Evaluation Results
The model was trained on the **T4 GPU** using Google Colab and evaluated on an independent test dataset (675 samples). 

| Metric | Score | Detail |
|---|---|---|
| **Overall Accuracy** | **63.11%** | 426 out of 675 test samples correctly predicted |
| **Fake Recall** | **73.05%** | Successfully detected **244 out of 334 actual fakes** |
| **Fake Precision** | **60.55%** | Out of all predicted fakes, **60.55%** were correct |
| **Real Recall** | **53.37%** | Correctly identified **182 out of 341 actual real samples** |

#### 📈 Training Curves & Confusion Matrix
Below are the training history curves and the final confusion matrix plots saved during evaluation:

<div align="center">
  <img src="./results/confusion_matrix_3000.png" width="48%" alt="Confusion Matrix" />
  <img src="./results/training_history_3000.png" width="48%" alt="Training History" />
</div>

---

## 🎯 How It Works

```
User uploads image/video
        ↓
MTCNN detects & extracts face
        ↓
Face resized to 224×224 & normalized
        ↓
EfficientNet-B4 predicts deepfake probability
        ↓
API returns verdict + confidence + details
        ↓
Frontend displays results with visual indicators
```

---

## 🛡️ Future Enhancements

- [ ] Real-time webcam detection
- [ ] Cloud deployment (AWS / HuggingFace Spaces)
- [ ] Docker containerization
- [ ] Improved model accuracy with newer architectures
- [ ] Batch processing support
- [ ] Explainability with GradCAM visualizations

---

## 👨‍💻 Author

**Atul Kumar**
GitHub: [@atul-kumar-30](https://github.com/atul-kumar-30)

---

## 📜 License

This project is built for **educational and research purposes** only.
Please use responsibly and ethically.
