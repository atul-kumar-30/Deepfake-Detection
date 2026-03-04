# 🔍 DeepFake Detection System

A full-stack DeepFake Detection web application built using **Flask, TensorFlow, MTCNN, and OpenCV**.  
The system analyzes uploaded images or videos and predicts whether the content is **REAL or FAKE** using deep learning techniques.

---

## 🚀 Features

- 🖼️ Image DeepFake Detection
- 🎥 Video DeepFake Detection (Frame Sampling)
- 🔍 Face Detection using MTCNN
- 🧠 EfficientNet-B4 Based Architecture
- 📊 Confidence Score & Probability Breakdown
- 🌐 REST API Backend (Flask)
- 💻 Interactive Frontend UI
- ⚡ Demo Mode Support (Mock Predictions if model not loaded)

---

## 🏗️ Tech Stack

**Backend**
- Flask
- Flask-CORS
- TensorFlow / Keras
- OpenCV
- MTCNN
- NumPy

**Frontend**
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

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/YOUR-USERNAME/YOUR-REPO-NAME.git
cd YOUR-REPO-NAME
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run Backend Server

```bash
python backend.py
```

Backend will run on:

```
http://localhost:5000
```

### 4️⃣ Open Frontend

Open:

```
index.html
```

in your browser.

---

## 📡 API Endpoints

### POST `/api/analyze`
Upload and analyze image/video.

Example using curl:

```bash
curl -X POST -F "file=@image.jpg" http://localhost:5000/api/analyze
```

### GET `/api/health`
Check backend health.

### GET `/api/model-info`
Get model metadata and status.

---

## 🧠 Model Information

- Architecture: EfficientNet-B4
- Face Detection: MTCNN
- Input Size: 224x224
- Supported Formats: JPG, PNG, MP4, AVI, MOV
- Training Datasets:
  - DFDC
  - FaceForensics++
  - Celeb-DF

> ⚠️ If trained model file is not present, system runs in **Demo Mode** with simulated predictions.

---

## 🎯 How It Works

1. User uploads image/video.
2. Face is extracted using MTCNN.
3. Face is resized and normalized.
4. Deep learning model predicts probability.
5. System returns:
   - REAL / FAKE verdict
   - Confidence score
   - Detailed analysis insights

---

## 📊 Example Response

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

## 🛡️ Future Improvements

- Real-time webcam detection
- Docker containerization
- Model optimization
- Deployment on cloud (AWS / Render / Heroku)

---

## 👨‍💻 Author

**Atul Kumar**

---

## 📜 License

This project is for educational and research purposes.
