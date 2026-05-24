import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import pickle

X = np.load("X.npy")
Y = np.load("Y.npy")

print(f"Data loaded! X: {X.shape}, Y: {Y.shape}")

le = LabelEncoder()
Y_encoded = le.fit_transform(Y)

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y_encoded, test_size=0.2, random_state=42, stratify=Y_encoded
)

X_train = torch.FloatTensor(X_train).unsqueeze(1)
X_test = torch.FloatTensor(X_test).unsqueeze(1)
Y_train = torch.LongTensor(Y_train)
Y_test = torch.LongTensor(Y_test)

train_dataset = TensorDataset(X_train, Y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

class EmotionCNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(EmotionCNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()
        
        conv_out_size = (input_size // 4) * 64
        
        self.fc1 = nn.Linear(conv_out_size, 128)
        self.fc2 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

input_size = X_train.shape[2]
num_classes = len(le.classes_)
model = EmotionCNN(input_size, num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("\nCNN Training shuru ho raha hai...")
print(f"Epochs: 50, Batch size: 32")
print("-" * 40)

for epoch in range(100):
    model.train()
    total_loss = 0
    for batch_X, batch_Y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_Y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if (epoch + 1) % 10 == 0:
        model.eval()
        with torch.no_grad():
            outputs = model(X_test)
            _, predicted = torch.max(outputs, 1)
            acc = accuracy_score(Y_test.numpy(), predicted.numpy())
        print(f"Epoch {epoch+1}/50 | Loss: {total_loss/len(train_loader):.4f} | Accuracy: {acc*100:.2f}%")

model.eval()
with torch.no_grad():
    outputs = model(X_test)
    _, predicted = torch.max(outputs, 1)
    final_acc = accuracy_score(Y_test.numpy(), predicted.numpy())

print(f"\nFinal Accuracy: {final_acc*100:.2f}%")

torch.save(model.state_dict(), "cnn_model.pth")

with open("cnn_label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

config = {"input_size": input_size, "num_classes": num_classes}
with open("cnn_config.pkl", "wb") as f:
    pickle.dump(config, f)

print("CNN Model save ho gaya — cnn_model.pth")