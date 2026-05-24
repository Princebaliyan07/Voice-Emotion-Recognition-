from flask import jsonify, request
from extract_features import extract_features
import os
from gtts import gTTS
from playsound import playsound

from flask import Flask, request, jsonify, render_template
import librosa
import numpy as np
import pickle
import os
import sounddevice as sd
import scipy.io.wavfile as wav

app = Flask(__name__)

with open("model.pkl", "rb") as f:
    model = pickle.load(f)
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

def speak_text(text):
    tts = gTTS(text=text, lang='en')
    filename = "output.mp3"
    tts.save(filename)
    playsound(filename)
    os.remove(filename)
    
def extract_features(file_path):
    audio, sr = librosa.load(file_path, duration=3, offset=0.5, sr=22050)
    mfcc = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(y=audio, sr=sr).T, axis=0)
    zcr = np.mean(librosa.feature.zero_crossing_rate(audio).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=audio, sr=sr).T, axis=0)
    return np.hstack([mfcc, mel, zcr, chroma])

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "audio" not in request.files:
            return jsonify({"error": "No file uploaded"})
        file = request.files["audio"]
        file_path = "temp_upload.wav"
        file.save(file_path)
        features = extract_features(file_path)
        features = scaler.transform([features])
        prediction = model.predict(features)
        emotion = le.inverse_transform(prediction)[0]
        if os.path.exists(file_path):
            os.remove(file_path)
        return jsonify({"emotion": emotion})
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route("/record", methods=["POST"])
def record():
    try:
        sample_rate = 22050
        audio = sd.rec(
            int(3 * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype='float32'
        )
        sd.wait()
        file_path = "live_recording.wav"
        wav.write(file_path, sample_rate, audio)
        features = extract_features(file_path)
        features = scaler.transform([features])
        prediction = model.predict(features)
        emotion = le.inverse_transform(prediction)[0]
        if os.path.exists(file_path):
            os.remove(file_path)
        return jsonify({"emotion": emotion})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True, threaded=True)