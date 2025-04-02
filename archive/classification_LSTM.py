import pandas as pd
import numpy as np
import re
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

# ---------------------------
# 1. Helper Function: Parse Sensor Reading String
# ---------------------------
def parse_reading(reading_str):
    """
    Extracts a list of floating point numbers from a string in the format 
    "[np.float64(117.90), np.float64(-35.40), ...]".
    """
    # Use regex to match numbers (including negatives and decimals)
    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", reading_str)
    return [float(num) for num in numbers]

# ---------------------------
# 2. Load and Preprocess Data
# ---------------------------
# Read the CSV file
df = pd.read_csv("./data/experiment_data.csv")

# Parse the "reading" column to convert the string to a list of floats
df["reading_parsed"] = df["reading"].apply(parse_reading)

# ---------------------------
# 3. Data Augmentation Using Sliding Window
# ---------------------------
# Settings: Each trial's original data has shape (20, 15).
# window_size is configurable; e.g., setting window_size = 16 means each window covers 16 time steps.
window_size = 16

augmented_data = []
labels_material = []
labels_motion = []
labels_material_motion = []
labels_static_slide = []

# Group data by material, motion, and trial, and sort each group by timestamp
grouped = df.groupby(["material", "motion", "trial"])
for (material, motion, trial), group in grouped:
    group_sorted = group.sort_values("timestamp")
    # Form a (trial_length, feature_dim) matrix for each trial
    trial_array = np.vstack(group_sorted["reading_parsed"].values)
    
    # Sliding window augmentation: number of windows = trial_length - window_size + 1
    for i in range(0, trial_array.shape[0] - window_size + 1):
        window = trial_array[i:i + window_size, :]  # shape: (window_size, 15)
        augmented_data.append(window)
        # Generate labels for four tasks
        labels_material.append(material)
        labels_motion.append(motion)
        labels_material_motion.append(f"{material}-{motion}")
        labels_static_slide.append("static" if motion == "tap" else "slide")

# Convert augmented data into numpy array
X = np.array(augmented_data)  # Shape: (total_windows, window_size, 15)
print("Shape of augmented data X:", X.shape)

# ---------------------------
# 4. Prepare Labels for Four Classification Tasks
# ---------------------------
tasks = {
    "Material": labels_material,
    "Motion": labels_motion,
    "Material-Motion": labels_material_motion,
    "Binary": labels_static_slide
}

# ---------------------------
# 5. Define a More Complex Network Architecture
# ---------------------------
def build_complex_model(num_classes, input_shape):
    """
    Builds a more complex model with two Conv1D layers (plus batch normalization and max pooling)
    followed by two bidirectional LSTM layers, a dense hidden layer, and dropout before the output.
    """
    model = tf.keras.models.Sequential([
        # Convolutional feature extractor
        tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        
        # Temporal modeling with bidirectional LSTM layers
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True)),
        #tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
        #tf.keras.layers.Dropout(0.2),
        
        # Fully connected layers
        tf.keras.layers.Dense(64, activation='relu'),
        #tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# ---------------------------
# 6. Train and Evaluate Models for Each Task
# ---------------------------
results_list = []  # To store the results for each task

for task_name, labels in tasks.items():
    print(f"\n--- Training for task: {task_name} ---")
    y = np.array(labels)
    
    # Label encoding and one-hot encoding
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    y_onehot = tf.keras.utils.to_categorical(y_encoded)
    num_classes = y_onehot.shape[1]
    print(f"Number of classes for {task_name}: {num_classes}")
    
    # Split data (stratified split)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_onehot, test_size=0.2, random_state=42, stratify=y_onehot
    )
    
    # Build a complex LSTM model
    model = build_complex_model(num_classes, input_shape=(window_size, X.shape[2]))
    model.summary()
    
    # Early stopping to prevent overfitting
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=False
    )
    
    history = model.fit(X_train, y_train,
                        validation_split=0.2,
                        epochs=20,
                        batch_size=32,
                        callbacks=[early_stopping],
                        verbose=1)
    
    # Evaluate on training and test sets
    train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Train Acc for task {task_name}: {train_acc:.4f}")
    print(f"Test Acc for task {task_name}: {test_acc:.4f}")
    
    # Compute precision, recall, and F1 score on the test set
    y_test_pred_prob = model.predict(X_test)
    y_test_pred = np.argmax(y_test_pred_prob, axis=1)
    y_test_true = np.argmax(y_test, axis=1)
    
    precision = precision_score(y_test_true, y_test_pred, average='macro', zero_division=0)
    recall = recall_score(y_test_true, y_test_pred, average='macro', zero_division=0)
    f1 = f1_score(y_test_true, y_test_pred, average='macro', zero_division=0)
    
    print(f"Precision for task {task_name}: {precision:.4f}")
    print(f"Recall for task {task_name}: {recall:.4f}")
    print(f"F1 Score for task {task_name}: {f1:.4f}")
    
    # Plot and save the loss curve for this task
    plt.figure()
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f"Loss Curve for {task_name}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f"loss_curve_{task_name.replace(' ', '_').replace('(', '').replace(')', '').replace('&','and')}.png")
    plt.close()
    
    # Save the metrics for this task
    results_list.append({
        "Task": task_name,
        "Train_Acc": round(train_acc, 4),
        "Test_Acc": round(test_acc, 4),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1_score": round(f1, 4)
    })

# ---------------------------
# 7. Output the Results Table
# ---------------------------
results_df = pd.DataFrame(results_list, columns=["Task", "Train_Acc", "Test_Acc", "Precision", "Recall", "F1_score"])
print("\nFinal Results:")
print(results_df)

# Save the results table as a CSV file (log file)
results_df.to_csv("classification_results.csv", index=False)
