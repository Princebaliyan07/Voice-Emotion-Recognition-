import sounddevice as sd
import numpy as np
import librosa
import pickle
from collections import deque

# Load model files
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

# Settings
SAMPLE_RATE = 22050
DURATION = 3  # seconds per prediction
BUFFER_SIZE = SAMPLE_RATE * DURATION

audio_buffer = deque(maxlen=BUFFER_SIZE)

# 🎯 Feature extraction (same improved version)
def extract_features_from_audio(audio, sr):
    mfcc = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(y=audio, sr=sr).T, axis=0)
    zcr = np.mean(librosa.feature.zero_crossing_rate(audio).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=audio, sr=sr).T, axis=0)

    rms = np.mean(librosa.feature.rms(y=audio).T, axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(y=audio, sr=sr).T, axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=audio, sr=sr).T, axis=0)

    return np.hstack([mfcc, mel, zcr, chroma, rms, contrast, tonnetz])

# 🎤 Audio callback (runs continuously)
def audio_callback(indata, frames, time, status):
    if status:
        print(status)
    
    audio_chunk = indata[:, 0]
    audio_buffer.extend(audio_chunk)

# 🚀 Start streaming
print("=" * 40)
print("🎤 Real-Time Emotion Detection Started")
print("Press Ctrl+C to stop")
print("=" * 40)

try:
    with sd.InputStream(callback=audio_callback,
                        channels=1,
                        samplerate=SAMPLE_RATE):

        while True:
            if len(audio_buffer) == BUFFER_SIZE:
                audio_np = np.array(audio_buffer)

                # Normalize
                audio_np = audio_np / np.max(np.abs(audio_np))

                # Extract features
                features = extract_features_from_audio(audio_np, SAMPLE_RATE)
                features = scaler.transform([features])

                # Predict
                prediction = model.predict(features)
                emotion = le.inverse_transform(prediction)[0]

                print(f"Detected Emotion: {emotion}")

except KeyboardInterrupt:
    print("\nStopped by user")