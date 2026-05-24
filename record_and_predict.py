import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import librosa
import pickle
import pyttsx3

engine = pyttsx3.init()

def speak_text(text):
    engine.say(text)
    engine.runAndWait()

with open("model.pkl", "rb") as f:
    model = pickle.load(f)
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

def extract_features(file_path):
    audio, sr = librosa.load(file_path, duration=3, offset=0.5, sr=22050)
    mfcc = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(y=audio, sr=sr).T, axis=0)
    zcr = np.mean(librosa.feature.zero_crossing_rate(audio).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=audio, sr=sr).T, axis=0)
    return np.hstack([mfcc, mel, zcr, chroma])

def record_and_predict():
    print("Recording... 3 seconds bolo!")
    sample_rate = 22050

    audio = sd.rec(int(3 * sample_rate),
                   samplerate=sample_rate,
                   channels=1,
                   dtype='float32')
    sd.wait()

    print("Recording complete!")

    wav.write("live_recording.wav", sample_rate, audio)

    features = extract_features("live_recording.wav")
    features = scaler.transform([features])
    prediction = model.predict(features)
    emotion = le.inverse_transform(prediction)[0]

    response_text = f"The detected emotion is {emotion}"
    speak_text(response_text)

    
    return jsonify({
    "emotion": emotion,
    "message": response_text
})

# LOOP
while True:
    input("\nPress Enter and speak...")
    emotion = record_and_predict()
    print(f"\nDetected Emotion: {emotion}")