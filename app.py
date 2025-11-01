from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import librosa
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Load the trained model
model = joblib.load("model.pkl")

def extract_features(file_path):
    """Extract features (MFCC only) from uploaded audio to match model training."""
    y, sr = librosa.load(file_path, sr=None)
    # Extract only 13 MFCC features + 2 additional features to match model's 15 features
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T, axis=0)
    # Add spectral centroid and rolloff to reach 15 features
    spec_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr).T, axis=0)
    spec_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr).T, axis=0)
    features = np.hstack([mfcc, spec_centroid, spec_rolloff])
    return features.reshape(1, -1)

@app.route('/')
def home():
    return render_template('login.html')

@app.route('/dashboard')
def dashboard():
    return render_template('home.html')

@app.route('/detect')
def detect():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/register')
def register():
    return render_template('register.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'audio' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files['audio']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    try:
        # Extract features & predict
        features = extract_features(file_path)
        prediction = model.predict(features)[0]
        confidence = model.predict_proba(features)[0].max()
        
        # Get waveform for visualization
        y, sr = librosa.load(file_path, sr=None)
        # Downsample for visualization (take every 1000th sample)
        waveform = y[::1000].tolist()[:200]  # Limit to 200 points
        
        if prediction == 1:
            label = "🚨 WHEEZE DETECTED - Consult Doctor"
        else:
            label = "✅ Normal Breathing Pattern"
        
        return jsonify({
            'result': label,
            'prediction': int(prediction),
            'confidence': float(confidence),
            'waveform': waveform
        })
        
    except Exception as e:
        return jsonify({'error': f'Processing failed: {str(e)}'})

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    app.run(debug=True)
