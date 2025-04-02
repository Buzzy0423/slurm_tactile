import os
import logging
import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ---------------------------
# Create necessary directories for logs and images
# ---------------------------
os.makedirs("log", exist_ok=True)
os.makedirs("imgs", exist_ok=True)
os.makedirs("models", exist_ok=True)

# ---------------------------
# Setup Logging
# ---------------------------
logging.basicConfig(
    filename='log/training_log_material.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger()

# ---------------------------
# 1. Reading and Preprocessing the Data
# ---------------------------
df = pd.read_csv("./data/material_classification_data.csv")
# Convert the 'reading' column from string to a list of floats.
df["reading"] = df["reading"].apply(ast.literal_eval)

# ---------------------------
# 2. Split Raw Data into Groups before Augmentation
# ---------------------------
# Group by both "material" and "trial"
grouped = df.groupby(["material", "trial"])
valid_groups = {}
for key, group in grouped:
    # Ensure each group is ordered by timestamp and has exactly 20 rows.
    group = group.sort_values("timestamp")
    if group.shape[0] == 20:
        valid_groups[key] = group
    else:
        logger.warning("Group (material: %s, trial: %s) has %d rows; skipping.", key[0], key[1], group.shape[0])

# Split the valid groups into training and test sets.
group_keys = list(valid_groups.keys())
train_keys, test_keys = train_test_split(group_keys, test_size=0.2, random_state=42)
logger.info("Total valid groups: %d, Training groups: %d, Test groups: %d", 
            len(group_keys), len(train_keys), len(test_keys))

# ---------------------------
# 3. Data Augmentation using a Sliding Window Along the Time Axis
# ---------------------------
def extract_windows_trial(trial_data, window_size=5, stride=1):
    """
    Extract sliding windows from trial data along the time axis.
    
    Parameters:
        trial_data (np.array): Array of shape (time_steps, num_features).
        window_size (int): Number of consecutive time steps in each window.
        stride (int): Step size for the sliding window.
    
    Returns:
        np.array: Array of shape (num_windows, window_size, num_features),
                  where num_windows = time_steps - window_size.
    """
    windows = []
    T = trial_data.shape[0]
    # For T=20 and window_size=5, yields 15 windows.
    for start in range(0, T - window_size, stride):
         window = trial_data[start:start + window_size, :]
         windows.append(window)
    return np.array(windows)

# Augment training data.
X_train_list = []
y_train_list = []
for key in train_keys:
    group = valid_groups[key]
    trial_data = np.stack(group["reading"].values)  # shape: (20, num_features)
    windows = extract_windows_trial(trial_data, window_size=5, stride=1)  # shape: (15, 5, num_features)
    X_train_list.append(windows)
    # key[0] is the material label.
    y_train_list.extend([key[0]] * windows.shape[0])
    
# Augment test data.
X_test_list = []
y_test_list = []
for key in test_keys:
    group = valid_groups[key]
    trial_data = np.stack(group["reading"].values)
    windows = extract_windows_trial(trial_data, window_size=5, stride=1)
    X_test_list.append(windows)
    y_test_list.extend([key[0]] * windows.shape[0])
    
# Concatenate windows from all groups.
X_train = np.concatenate(X_train_list, axis=0)
X_test = np.concatenate(X_test_list, axis=0)
y_train = np.array(y_train_list)
y_test = np.array(y_test_list)

logger.info("Augmented training data shape: %s", X_train.shape)  # Expected: (num_train_windows, 5, num_features)
logger.info("Augmented test data shape: %s", X_test.shape)
logger.info("Training labels shape: %s, Test labels shape: %s", y_train.shape, y_test.shape)

# ---------------------------
# 4. Encode Material Labels and Build the LSTM Model
# ---------------------------
# Encode the material labels (e.g., "metal", "plastic", "wood") into integer labels.
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.transform(y_test)
num_classes = len(le.classes_)
logger.info("Encoded material classes: %s", le.classes_)

num_features = X_train.shape[2]  # Typically 15

# Build the LSTM-based classifier.
model = Sequential()
model.add(LSTM(64, input_shape=(5, num_features)))
model.add(Dense(32, activation='relu'))  # Adding a dense layer for better learning
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# ---------------------------
# 5. Train and Evaluate the Model
# ---------------------------
epochs = 50
batch_size = 16
logger.info("Training Parameters: epochs=%d, batch_size=%d", epochs, batch_size)
logger.info("Train dataset size: %d samples", len(X_train))
logger.info("Test dataset size: %d samples", len(X_test))

history = model.fit(X_train, y_train_encoded, epochs=epochs, batch_size=batch_size, validation_split=0.1)

# Save the trained model.
model_path = "models/material_classifier.keras"
model.save(model_path)
logger.info("Model saved to %s", model_path)

# ---------------------------
# 6. Plot Training and Test Loss Curves
# ---------------------------
plt.figure(figsize=(8, 6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title("Training and Test Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
loss_plot_path = "imgs/loss_curve_material.png"
plt.savefig(loss_plot_path, dpi=300)  # Save the figure with high resolution
plt.close()
logger.info("Loss plot saved to %s", loss_plot_path)

# ---------------------------
# 7. Log Detailed Epoch Metrics
# ---------------------------
for epoch in range(epochs):
    logger.info(
        "Epoch %d: Train Loss: %.4f, Train Accuracy: %.4f, Val Loss: %.4f, Val Accuracy: %.4f",
        epoch + 1,
        history.history['loss'][epoch],
        history.history['accuracy'][epoch],
        history.history['val_loss'][epoch],
        history.history['val_accuracy'][epoch]
    )

test_loss, test_accuracy = model.evaluate(X_test, y_test_encoded)
logger.info("Final Test Loss: %.4f, Test Accuracy: %.4f", test_loss, test_accuracy)