"""
dl_model.py  –  Ambient Assisted Living: DL Classifier + Autoencoder
=====================================================================
UCI HAR Dataset  (Human Activity Recognition using Smartphones)

Trains:
  1. A dense classifier  →  predicts 6 activity classes
  2. An autoencoder      →  detects anomalous sensor readings

Saves:
  classifier_model.h5
  autoencoder_model.h5

Then run  llm_report_generator.py  to produce a PDF report.

NOTE: Every 1000 rows of test data = one user session.
"""

import os
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (Dense, Dropout, BatchNormalization,
                                     Input)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ─────────────────────────────────────────────────────────────────────────────
# PATHS  – change BASE_DIR if needed, everything else is relative
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "..", "UCI HAR Dataset")

X_TRAIN_PATH = os.path.join(DATASET_DIR, "train", "X_train.txt")
Y_TRAIN_PATH = os.path.join(DATASET_DIR, "train", "y_train.txt")
X_TEST_PATH  = os.path.join(DATASET_DIR, "test",  "X_test.txt")
Y_TEST_PATH  = os.path.join(DATASET_DIR, "test",  "y_test.txt")

SESSION_SIZE = 1000          # rows per user session


# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
print("Loading data …")
X_train     = np.loadtxt(X_TRAIN_PATH)
y_train_raw = np.loadtxt(Y_TRAIN_PATH).astype(int) - 1   # 0-indexed

X_test      = np.loadtxt(X_TEST_PATH)
y_test_raw  = np.loadtxt(Y_TEST_PATH).astype(int) - 1

print(f"X_train: {X_train.shape}   X_test: {X_test.shape}")

y_train = to_categorical(y_train_raw, num_classes=6)
y_test  = to_categorical(y_test_raw,  num_classes=6)

INPUT_DIM   = X_train.shape[1]   # 561
LATENT_DIM  = 32


# ─────────────────────────────────────────────────────────────────────────────
# 2. CLASSIFIER
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Training Classifier ──")

classifier = Sequential([
    Dense(512, activation="relu", input_shape=(INPUT_DIM,)),
    BatchNormalization(),
    Dropout(0.4),

    Dense(256, activation="relu"),
    BatchNormalization(),
    Dropout(0.3),

    Dense(128, activation="relu"),
    BatchNormalization(),
    Dropout(0.2),

    Dense(64, activation="relu"),
    BatchNormalization(),

    Dense(6, activation="softmax")
], name="activity_classifier")

classifier.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)
classifier.summary()

callbacks_cls = [
    EarlyStopping(patience=5, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(factor=0.5, patience=3, verbose=1)
]

history_cls = classifier.fit(
    X_train, y_train,
    epochs=50,
    batch_size=64,
    validation_split=0.15,
    callbacks=callbacks_cls,
    verbose=1
)

loss_cls, acc_cls = classifier.evaluate(X_test, y_test, verbose=0)
print(f"\nClassifier  →  Test Accuracy: {acc_cls:.4f}   Loss: {loss_cls:.4f}")

classifier.save(os.path.join(BASE_DIR, "classifier_model.h5"))
print("Saved: classifier_model.h5")


# ─────────────────────────────────────────────────────────────────────────────
# 3. AUTOENCODER  (anomaly detection)
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Training Autoencoder ──")

ae_input = Input(shape=(INPUT_DIM,), name="ae_input")

# Encoder
x = Dense(256, activation="relu")(ae_input)
x = BatchNormalization()(x)
x = Dense(128, activation="relu")(x)
x = BatchNormalization()(x)
latent = Dense(LATENT_DIM, activation="relu", name="latent")(x)

# Decoder
x = Dense(128, activation="relu")(latent)
x = Dense(256, activation="relu")(x)
ae_output = Dense(INPUT_DIM, activation="linear", name="ae_output")(x)

autoencoder = Model(ae_input, ae_output, name="autoencoder")
encoder     = Model(ae_input, latent,    name="encoder")

autoencoder.compile(optimizer="adam", loss="mse")
autoencoder.summary()

callbacks_ae = [
    EarlyStopping(patience=5, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(factor=0.5, patience=3, verbose=1)
]

history_ae = autoencoder.fit(
    X_train, X_train,
    epochs=50,
    batch_size=64,
    validation_split=0.15,
    callbacks=callbacks_ae,
    verbose=1
)

autoencoder.save(os.path.join(BASE_DIR, "autoencoder_model.h5"))
print("Saved: autoencoder_model.h5")


# ─────────────────────────────────────────────────────────────────────────────
# 4. QUICK INFERENCE PREVIEW  (first 5 test samples)
# ─────────────────────────────────────────────────────────────────────────────
ACTIVITY_NAMES = {
    0: "Walking", 1: "Walking Upstairs", 2: "Walking Downstairs",
    3: "Sitting", 4: "Standing", 5: "Laying"
}

reconstructed       = autoencoder.predict(X_test, verbose=0)
reconstruction_error= np.mean(np.square(X_test - reconstructed), axis=1)
threshold           = np.percentile(reconstruction_error, 95)
anomaly_flags       = reconstruction_error > threshold

y_pred              = classifier.predict(X_test, verbose=0)
activity_classes    = np.argmax(y_pred, axis=1)
confidence_scores   = np.max(y_pred, axis=1)
latent_vectors      = encoder.predict(X_test, verbose=0)

print(f"\nAnomaly threshold (95th pct): {threshold:.5f}")
print(f"Total anomalies detected    : {np.sum(anomaly_flags)}")
print(f"Latent vector shape         : {latent_vectors.shape}")

print("\n── Sample Predictions ─────────────────────────────────────────────────")
for i in range(5):
    print(f"\n  Sample {i:>4d}")
    print(f"    True Activity     : {ACTIVITY_NAMES.get(y_test_raw[i], '?')}")
    print(f"    Predicted Activity: {ACTIVITY_NAMES.get(activity_classes[i], '?')}")
    print(f"    Confidence        : {confidence_scores[i]:.3f}")
    print(f"    Reconstruction Err: {reconstruction_error[i]:.5f}")
    print(f"    Anomalous         : {anomaly_flags[i]}")
    print(f"    Latent (first 5)  : {np.round(latent_vectors[i][:5], 3)}")

# Session-level summary
n_sessions = len(X_test) // SESSION_SIZE
print(f"\n── Session Summary ({SESSION_SIZE} rows / session) ──────────────────────")
for s in range(n_sessions):
    start, end = s * SESSION_SIZE, (s + 1) * SESSION_SIZE
    acts   = activity_classes[start:end]
    vals, counts = np.unique(acts, return_counts=True)
    dominant = vals[np.argmax(counts)]
    n_anom   = np.sum(anomaly_flags[start:end])
    print(f"  Session {s+1:>3d} | dominant={ACTIVITY_NAMES[dominant]:<22s} "
          f"| anomalies={n_anom:>4d}")

print("\n[DONE] Run llm_report_generator.py to generate the PDF report.")