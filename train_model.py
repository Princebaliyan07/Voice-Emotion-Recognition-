import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import pickle

X = np.load("X.npy")
Y = np.load("Y.npy")

print(f"Data loaded! X: {X.shape}, Y: {Y.shape}")

scaler = StandardScaler()
X = scaler.fit_transform(X)

le = LabelEncoder()
Y_encoded = le.fit_transform(Y)

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y_encoded, test_size=0.2, random_state=42, stratify=Y_encoded
)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

print("\nModel train ho raha hai...")
model = SVC(kernel='rbf', C=100, gamma='scale', 
            random_state=42, class_weight='balanced')
model.fit(X_train, Y_train)

Y_pred = model.predict(X_test)
accuracy = accuracy_score(Y_test, Y_pred)

print(f"\nModel train ho gaya!")
print(f"Accuracy: {accuracy*100:.2f}%")
print("\nDetailed Report:")
print(classification_report(Y_test, Y_pred, 
      target_names=le.classes_))

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

print("\nModel save ho gaya — model.pkl")