# 🔍 DeepFake Detection System

A full-stack DeepFake Detection web application built using **Flask, TensorFlow, MTCNN, and OpenCV**.

This system analyzes uploaded **images and videos** and predicts whether the content is **REAL or FAKE** using deep learning-based facial analysis.

---

## 🚀 Live Project Overview

The application consists of:

- 🧠 Deep Learning Backend (Flask API)
- 🖼️ Image & 🎥 Video Analysis Support
- 🔍 Face Detection using MTCNN
- 📊 Confidence Score with Probability Breakdown
- 🌐 Interactive Frontend UI
- ⚡ Demo Mode (if trained model not loaded)

---

## 🏗️ Tech Stack

### 🔹 Backend
- Flask
- Flask-CORS
- TensorFlow / Keras
- OpenCV
- MTCNN
- NumPy

### 🔹 Frontend
- HTML
- CSS
- JavaScript (Fetch API)

---

## 📂 Project Structure

```
Deepfake-Detection/
│
├── backend.py
├── index.html
├── requirements.txt
├── uploads/
└── README.md
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone Repository

```bash
git clone https://github.com/atul-kumar-30/Deepfake-Detection.git
cd Deepfake-Detection
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run Backend Server

```bash
python backend.py
```

Backend runs at:

```
http://localhost:5000
```

### 4️⃣ Open Frontend

Open `index.html` in your browser.

Upload an image or video to analyze.

---

## 📡 API Endpoints

### 🔹 POST `/api/analyze`
Upload and analyze media file.

Example:

```bash
curl -X POST -F "file=@image.jpg" http://localhost:5000/api/analyze
```

### 🔹 GET `/api/health`
Check backend status.

### 🔹 GET `/api/model-info`
Retrieve model metadata and loading status.

---

## 🧠 Model Details

- Architecture: EfficientNet-B4
- Face Detection: MTCNN
- Input Size: 224x224
- Supported Formats: JPG, PNG, MP4, AVI, MOV
- Training Datasets:
  - DFDC
  - FaceForensics++
  - Celeb-DF

> ⚠️ If trained `.h5` model file is not found, system runs in **Demo Mode** with simulated predictions.

---

## 🎯 How It Works

1. User uploads image or video.
2. MTCNN extracts the face.
3. Face is resized and normalized.
4. Model predicts deepfake probability.
5. API returns:
   - REAL / FAKE verdict
   - Confidence percentage
   - Detailed analysis indicators

---

## 📊 Example API Response

```json
{
  "success": true,
  "is_fake": false,
  "confidence": 87.4,
  "fake_probability": 12.6,
  "authentic_probability": 87.4
}
```

---

## 🛡️ Future Enhancements

- Real-time webcam detection
- Model deployment on cloud
- Docker containerization
- Improved deepfake detection accuracy
- Frontend UI enhancements

---

## 👨‍💻 Author

**Atul Kumar**  
GitHub: https://github.com/atul-kumar-30

---

## 📜 License

This project is built for educational and research purposes.
