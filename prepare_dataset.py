import librosa
import numpy as np
import os

def extract_features(file_path):
    audio, sr = librosa.load(file_path, duration=3, offset=0.5)
    mfcc = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(y=audio, sr=sr).T, axis=0)
    zcr = np.mean(librosa.feature.zero_crossing_rate(audio).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=audio, sr=sr).T, axis=0)
    return np.hstack([mfcc, mel, zcr, chroma])

emotions = {
    1: "Neutral", 2: "Calm", 3: "Happy", 4: "Sad",
    5: "Angry", 6: "Fearful", 7: "Disgust", 8: "Surprised"
}

X = []
Y = []

data_path = "data"
count = 0

for actor in os.listdir(data_path):
    actor_path = os.path.join(data_path, actor)
    for file in os.listdir(actor_path):
        if file.endswith(".wav"):
            file_path = os.path.join(actor_path, file)
            emotion_code = int(file.split("-")[2])
            emotion = emotions[emotion_code]
            features = extract_features(file_path)
            X.append(features)
            Y.append(emotion)
            count += 1
            print(f"Processing: {count} - {file} - {emotion}")

X = np.array(X)
Y = np.array(Y)

np.save("X.npy", X)
np.save("Y.npy", Y)

print(f"\nDataset ready!")
print(f"Total samples: {len(X)}")
print(f"Feature vector size: {X.shape}")