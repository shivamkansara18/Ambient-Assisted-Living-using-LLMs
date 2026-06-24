import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# ==============================
# DEVICE
# ==============================
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

device = get_device()

# ==============================
# PATHS
# ==============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "ucihar")

X_TRAIN_PATH = os.path.join(DATASET_DIR, "train", "X_train.txt")
Y_TRAIN_PATH = os.path.join(DATASET_DIR, "train", "y_train.txt")
X_TEST_PATH  = os.path.join(DATASET_DIR, "test",  "X_test.txt")
Y_TEST_PATH  = os.path.join(DATASET_DIR, "test",  "y_test.txt")

SESSION_SIZE = 1000

# ==============================
# LOAD DATA
# ==============================
print("Loading data...")
X_train = np.loadtxt(X_TRAIN_PATH)
y_train = np.loadtxt(Y_TRAIN_PATH).astype(int) - 1

X_test = np.loadtxt(X_TEST_PATH)
y_test = np.loadtxt(Y_TEST_PATH).astype(int) - 1

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)

X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

INPUT_DIM = X_train.shape[1]
LATENT_DIM = 32

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
test_loader  = DataLoader(TensorDataset(X_test, y_test), batch_size=64)

# ==============================
# CLASSIFIER MODEL
# ==============================
class Classifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.4),

            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),

            nn.Linear(64, 6)
        )

    def forward(self, x):
        return self.net(x)

classifier = Classifier(INPUT_DIM).to(device)

criterion_cls = nn.CrossEntropyLoss()
optimizer_cls = optim.Adam(classifier.parameters(), lr=1e-3)

# ==============================
# TRAIN CLASSIFIER
# ==============================
print("\nTraining Classifier...")

for epoch in range(50):
    classifier.train()
    total_loss = 0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        optimizer_cls.zero_grad()
        outputs = classifier(x)
        loss = criterion_cls(outputs, y)
        loss.backward()
        optimizer_cls.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}: Loss = {total_loss:.4f}")

# ==============================
# EVALUATE CLASSIFIER
# ==============================
classifier.eval()
correct = 0
total = 0

with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        outputs = classifier(x)
        preds = torch.argmax(outputs, dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

print(f"\nClassifier Accuracy: {correct / total:.4f}")

torch.save(classifier.state_dict(), os.path.join(BASE_DIR, "classifier_model.pth"))

# ==============================
# AUTOENCODER
# ==============================
class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),

            nn.Linear(128, latent_dim),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),

            nn.Linear(128, 256),
            nn.ReLU(),

            nn.Linear(256, input_dim)
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent

autoencoder = Autoencoder(INPUT_DIM, LATENT_DIM).to(device)

criterion_ae = nn.MSELoss()
optimizer_ae = optim.Adam(autoencoder.parameters(), lr=1e-3)

# ==============================
# TRAIN AUTOENCODER
# ==============================
print("\nTraining Autoencoder...")

for epoch in range(50):
    autoencoder.train()
    total_loss = 0

    for x, _ in train_loader:
        x = x.to(device)

        optimizer_ae.zero_grad()
        reconstructed, _ = autoencoder(x)
        loss = criterion_ae(reconstructed, x)
        loss.backward()
        optimizer_ae.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}: Loss = {total_loss:.4f}")

torch.save(autoencoder.state_dict(), os.path.join(BASE_DIR, "autoencoder_model.pth"))

# ==============================
# INFERENCE
# ==============================
print("\nRunning inference...")

autoencoder.eval()
classifier.eval()

with torch.no_grad():
    reconstructed, latent = autoencoder(X_test.to(device))
    reconstruction_error = torch.mean((X_test.to(device) - reconstructed) ** 2, dim=1)

    threshold = torch.quantile(reconstruction_error, 0.95)
    anomalies = reconstruction_error > threshold

    outputs = classifier(X_test.to(device))
    probs = torch.softmax(outputs, dim=1)
    preds = torch.argmax(probs, dim=1)

print(f"\nAnomaly threshold: {threshold.item():.5f}")
print(f"Total anomalies: {anomalies.sum().item()}")

# ==============================
# SAMPLE OUTPUT
# ==============================
ACTIVITY_NAMES = {
    0: "Walking", 1: "Walking Upstairs", 2: "Walking Downstairs",
    3: "Sitting", 4: "Standing", 5: "Laying"
}

for i in range(5):
    print(f"\nSample {i}")
    print(f"True: {ACTIVITY_NAMES[y_test[i].item()]}")
    print(f"Pred: {ACTIVITY_NAMES[preds[i].item()]}")
    print(f"Conf: {probs[i].max().item():.3f}")
    print(f"Recon Error: {reconstruction_error[i].item():.5f}")
    print(f"Anomaly: {anomalies[i].item()}")
    print(f"Latent[:5]: {latent[i][:5].cpu().numpy()}")