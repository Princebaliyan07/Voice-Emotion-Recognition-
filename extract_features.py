import librosa
import numpy as np

def extract_features(file_path):
    audio, sr = librosa.load(file_path, duration=3, offset=0.5)
    
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    mfcc_mean = np.mean(mfcc.T, axis=0)
    
    mel = librosa.feature.melspectrogram(y=audio, sr=sr)
    mel_mean = np.mean(mel.T, axis=0)
    
    zcr = librosa.feature.zero_crossing_rate(audio)
    zcr_mean = np.mean(zcr.T, axis=0)
    
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    chroma_mean = np.mean(chroma.T, axis=0)
    
    features = np.hstack([mfcc_mean, mel_mean, zcr_mean, chroma_mean])
    
    return features

file_path = "data/Actor_01/03-01-01-01-01-01-01.wav"
features = extract_features(file_path)

print(f"Features extract ho gayi!")
print(f"Feature vector shape: {features.shape}")
print(f"Total features: {len(features)}")