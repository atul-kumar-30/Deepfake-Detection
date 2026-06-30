from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS
import os
import numpy as np
import cv2
from werkzeug.utils import secure_filename
import tensorflow as tf
from mtcnn import MTCNN
import subprocess
import shutil
import uuid
import threading
import base64

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
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidate_paths = [
        os.path.join(script_dir, "deepfake_best_3000samples.h5"),
        os.path.join(script_dir, "models", "deepfake_best_3000samples.h5"),
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


def get_verdict(prediction):
    if prediction > 0.5:
        return True, "FAKE"
    else:
        return False, "REAL"


def adjust_scores(prediction, verdict):
    fake = round(prediction * 100, 1)
    authentic = round((1 - prediction) * 100, 1)
    return authentic, fake


def analyze_image(path):
    img = cv2.imread(path)
    if img is None:
        return {'success': False, 'error': 'Invalid image'}

    if model is not None:
        face = extract_face(img)
        if face is None:
            return {'success': False, 'error': 'No face detected in the image'}
        face = face.astype('float32') / 255.0
        face = np.expand_dims(face, axis=0)
        prediction = float(model.predict(face, verbose=0)[0][0])
    else:
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


def analyze_video(path):
    cap = cv2.VideoCapture(path)
    preds = []

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_count = min(30, total)
    indices = np.linspace(0, total - 1, sample_count, dtype=int)

    if model is not None:
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
        'demo_mode': model is None
    }


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


@app.route('/api/thumbnail', methods=['POST'])
def get_thumbnail():
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file uploaded'})

    file = request.files['file']
    if not file or file.filename == '':
        return jsonify({'success': False, 'error': 'Empty filename'})

    filename = secure_filename(file.filename)
    path = os.path.join(UPLOAD_FOLDER, 'thumb_' + filename)
    file.save(path)

    try:
        cap = cv2.VideoCapture(path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        mid = max(0, total // 2)
        cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
        ret, frame = cap.read()
        cap.release()

        if not ret or frame is None:
            return jsonify({'success': False, 'error': 'Could not read video frame'})

        h, w = frame.shape[:2]
        max_dim = 640
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        b64 = base64.b64encode(buffer).decode('utf-8')
        return jsonify({'success': True, 'thumbnail': f'data:image/jpeg;base64,{b64}'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})
    finally:
        if os.path.exists(path):
            os.remove(path)


@app.route('/api/convert-video', methods=['POST'])
def convert_video():
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file uploaded'})
    file = request.files['file']
    if not file or file.filename == '':
        return jsonify({'success': False, 'error': 'Empty filename'})

    WINGET_FFMPEG = r"C:\Users\atulk\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1.2-full_build\bin\ffmpeg.exe"
    ffmpeg_path = shutil.which('ffmpeg') or (WINGET_FFMPEG if os.path.exists(WINGET_FFMPEG) else None)
    if not ffmpeg_path:
        return jsonify({'success': False, 'error': 'ffmpeg not found'})

    filename = secure_filename(file.filename)
    src_path = os.path.join(UPLOAD_FOLDER, 'src_' + filename)
    uid = uuid.uuid4().hex
    out_path = os.path.join(UPLOAD_FOLDER, f'play_{uid}.mp4')
    file.save(src_path)

    try:
        result = subprocess.run(
            [ffmpeg_path, '-y', '-i', src_path,
             '-vcodec', 'libx264', '-acodec', 'aac',
             '-crf', '28', '-preset', 'ultrafast',
             '-movflags', 'faststart', out_path],
            capture_output=True, timeout=120
        )
        if result.returncode != 0:
            return jsonify({'success': False, 'error': 'ffmpeg conversion failed'})

        def cleanup():
            import time
            time.sleep(300)
            if os.path.exists(out_path):
                os.remove(out_path)
        threading.Thread(target=cleanup, daemon=True).start()

        return jsonify({'success': True, 'video_id': uid})
    except subprocess.TimeoutExpired:
        return jsonify({'success': False, 'error': 'Conversion timed out'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})
    finally:
        if os.path.exists(src_path):
            os.remove(src_path)


@app.route('/api/serve-video/<video_id>')
def serve_video(video_id):
    if not all(c in '0123456789abcdef' for c in video_id):
        return jsonify({'error': 'Invalid video ID'}), 400
    path = os.path.join(UPLOAD_FOLDER, f'play_{video_id}.mp4')
    if not os.path.exists(path):
        return jsonify({'error': 'Video not found or expired'}), 404
    return send_file(path, mimetype='video/mp4', conditional=True)


@app.route('/api/health')
def health():
    return jsonify({'status': 'ok', 'model_loaded': model is not None})


@app.route('/')
def serve_frontend():
    return send_from_directory(BASE_DIR, 'frontend.html')


if __name__ == '__main__':
    init_detector()
    load_model()
    app.run(debug=True)