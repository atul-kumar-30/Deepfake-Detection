from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import numpy as np
import cv2
from werkzeug.utils import secure_filename
import tensorflow as tf
from mtcnn import MTCNN

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'mp4', 'avi', 'mov'}
IMG_SIZE = 224

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

model = None
face_detector = None


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def load_model():
    global model
    # Look for the model in common locations relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidate_paths = [
        os.path.join(script_dir, "deepfake_best_3000samples.h5"),
        os.path.join(script_dir, "models", "deepfake_best_3000samples.h5"),
        r"c:\Users\Ayush\models\deepfake_best_3000samples.h5",
    ]
    for path in candidate_paths:
        if os.path.exists(path):
            try:
                model = tf.keras.models.load_model(path)
                print("[OK] Model loaded from:", path)
                return True
            except Exception as e:
                print("[WARN] Failed to load model:", e)
    print("[INFO] Model not found -> running in demo mode (random predictions)")
    return False


def init_detector():
    global face_detector
    face_detector = MTCNN()


def extract_face(image):
    try:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        faces = face_detector.detect_faces(image)

        if len(faces) > 0:
            face = max(faces, key=lambda x: x['box'][2] * x['box'][3])
            x, y, w, h = face['box']

            # Add 20% margin around face for better context
            margin_x = int(w * 0.2)
            margin_y = int(h * 0.2)
            x = max(0, x - margin_x)
            y = max(0, y - margin_y)
            w = min(image.shape[1] - x, w + 2 * margin_x)
            h = min(image.shape[0] - y, h + 2 * margin_y)

            face_img = image[y:y+h, x:x+w]
            face_img = cv2.resize(face_img, (IMG_SIZE, IMG_SIZE))

            return face_img

        return None
    except:
        return None


# ─────────────────────────────────────────────────────────────
# DECISION LOGIC
#
# FIX: The old thresholds (>0.75 FAKE, <0.55 REAL, else UNCERTAIN)
# created a large dead-zone where borderline fakes were silently
# treated as "not fake" (is_fake=False). This caused real fake
# videos to slip through as REAL in the frontend.
#
# New logic:
#   prediction > 0.50  → FAKE   (model thinks fake)
#   prediction <= 0.50 → REAL   (model thinks real)
#   confidence < 30%   → mark UNCERTAIN regardless of direction
#
# This makes the boundary match the model's actual sigmoid output,
# while UNCERTAIN now means "I'm not sure either way" rather than
# silently defaulting to REAL.
# ─────────────────────────────────────────────────────────────
def get_verdict(prediction):
    # Clean 0.5 boundary. UNCERTAIN removed because with a small training
    # set (3000 samples) the model's sigmoid outputs cluster near 0.5 and
    # the old confidence < 30% zone was swallowing almost every result.
    if prediction > 0.5:
        return True, "FAKE"
    else:
        return False, "REAL"


# SCORE ALIGNMENT
def adjust_scores(prediction, verdict):
    if verdict == "REAL":
        authentic = max(55, (1 - prediction) * 100)
        fake = 100 - authentic
    else:  # FAKE
        fake = max(55, prediction * 100)
        authentic = 100 - fake

    return round(authentic, 1), round(fake, 1)


# ================= IMAGE =================

def analyze_image(path):
    img = cv2.imread(path)
    if img is None:
        return {'success': False, 'error': 'Invalid image'}

    if model is not None:
        # Real model: requires face detection
        face = extract_face(img)
        if face is None:
            return {'success': False, 'error': 'No face detected in the image'}
        face = face.astype('float32') / 255.0
        face = np.expand_dims(face, axis=0)
        prediction = float(model.predict(face, verbose=0)[0][0])
    else:
        # Demo mode: skip face detection, return random prediction
        prediction = float(np.random.uniform(0.2, 0.85))

    is_fake, verdict = get_verdict(prediction)
    authentic, fake = adjust_scores(prediction, verdict)
    confidence = round(abs(prediction - 0.5) * 2 * 100, 1)

    return {
        'success': True,
        'is_fake': is_fake,
        'verdict': verdict,
        'confidence': confidence,
        'authentic_probability': authentic,
        'fake_probability': fake,
        'raw_score': round(prediction, 4),
        'demo_mode': model is None
    }


# ================= VIDEO =================

def analyze_video(path):
    cap = cv2.VideoCapture(path)
    preds = []

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_count = min(30, total)
    indices = np.linspace(0, total - 1, sample_count, dtype=int)

    if model is not None:
        # Real model: run MTCNN on sampled frames
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if not ret:
                continue
            face = extract_face(frame)
            if face is not None:
                face = face.astype('float32') / 255.0
                face = np.expand_dims(face, axis=0)
                p = float(model.predict(face, verbose=0)[0][0])
                preds.append(p)
        cap.release()
        if not preds:
            return {'success': False, 'error': 'No face detected in video'}
    else:
        # Demo mode: generate random predictions per sampled frame
        cap.release()
        count = max(1, sample_count)
        preds = [float(np.random.uniform(0.2, 0.85)) for _ in range(count)]

    avg_pred = float(np.mean(preds))
    fake_frame_ratio = sum(1 for p in preds if p > 0.5) / len(preds)

    is_fake, verdict = get_verdict(avg_pred)
    authentic, fake = adjust_scores(avg_pred, verdict)
    confidence = round(abs(avg_pred - 0.5) * 2 * 100, 1)

    return {
        'success': True,
        'is_fake': is_fake,
        'verdict': verdict,
        'confidence': confidence,
        'authentic_probability': authentic,
        'fake_probability': fake,
        'frames_analyzed': len(preds),
        'fake_frame_ratio': round(fake_frame_ratio * 100, 1),
        'raw_score': round(avg_pred, 4),
        'demo_mode': True
    }


# ================= API =================

@app.route('/api/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file uploaded'})

    file = request.files['file']
    if not file or file.filename == '':
        return jsonify({'success': False, 'error': 'Empty filename'})

    filename = secure_filename(file.filename)
    path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(path)

    ext = filename.rsplit('.', 1)[-1].lower()

    try:
        if ext in ['jpg', 'jpeg', 'png']:
            result = analyze_image(path)
        elif ext in ['mp4', 'avi', 'mov']:
            result = analyze_video(path)
        else:
            result = {'success': False, 'error': 'Unsupported file type'}
    finally:
        if os.path.exists(path):
            os.remove(path)

    return jsonify(result)


@app.route('/api/health')
def health():
    return jsonify({'status': 'ok', 'model_loaded': model is not None})


@app.route('/')
def serve_frontend():
    return send_from_directory(BASE_DIR, 'frontend.html')


# ================= RUN =================

if __name__ == '__main__':
    init_detector()
    load_model()
    app.run(debug=True)