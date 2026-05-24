import librosa
import numpy as np
import os

data_path = "data"

file_path = "data/Actor_01/03-01-01-01-01-01-01.wav"

audio, sample_rate = librosa.load(file_path, duration=3)

print(f"File load ho gayi!")
print(f"Audio length: {len(audio)} samples")
print(f"Sample rate: {sample_rate} Hz")
print(f"Duration: {len(audio)/sample_rate:.1f} seconds")

filename = os.path.basename(file_path)
emotion_code = int(filename.split("-")[2])

emotions = {
    1: "Neutral",
    2: "Calm", 
    3: "Happy",
    4: "Sad",
    5: "Angry",
    6: "Fearful",
    7: "Disgust",
    8: "Surprised"
}

print(f"Emotion: {emotions[emotion_code]}")