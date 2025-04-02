import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def parse_reading(s):
    """
    Parse a string representing a list of np.float64 values into a Python list of floats.
    For example, converts:
    "[np.float64(117.9000015258789), np.float64(-35.400001525878906), ...]"
    into:
    [117.9000015258789, -35.4000015258789, ...]
    """
    s_clean = s.replace("np.float64(", "").replace(")", "")
    try:
        return ast.literal_eval(s_clean)
    except Exception as e:
        print("Error parsing reading:", s)
        raise e

def flatten_trial(trial_readings):
    """
    Flatten a list of sensor readings into a single feature vector.
    Each sensor reading is assumed to be a list or array.
    """
    return np.concatenate(trial_readings)

def load_and_process_data(csv_file, window_size=10):
    """
    Load the CSV file and augment the dataset using a sliding window.
    
    For each trial (grouped by material, motion, trial number), assume the sensor 
    collected 20 samples. Then, for each trial, generate augmented samples by taking 
    a sliding window of `window_size` continuous readings. The number of windows is 
    calculated as (len(readings) - window_size + 1).
    
    Returns:
        X: Numpy array of shape (n_augmented_samples, window_size * sensor_dim)
        labels_material: List of material labels for each augmented sample
        labels_motion: List of operation (motion) labels for each augmented sample
        labels_joint: List of joint labels (material_operation) for each augmented sample
        labels_binary: List of binary labels ("tap" vs "rub_push") for each augmented sample
    """
    df = pd.read_csv(csv_file)
    df['reading'] = df['reading'].apply(parse_reading)
    
    grouped = df.groupby(['material', 'motion', 'trial'])
    
    features = []
    labels_material = []
    labels_motion = []
    labels_joint = []
    labels_binary = []
    
    for name, group in grouped:
        # name: (material, motion, trial)
        group = group.sort_values('timestamp')
        readings = group['reading'].tolist()
        
        if len(readings) < window_size:
            continue
        
        num_windows = len(readings) - window_size + 1
        for i in range(num_windows):
            window = readings[i:i + window_size]
            feature_vector = flatten_trial(window)
            features.append(feature_vector)
            material, motion, trial = name
            labels_material.append(material)
            labels_motion.append(motion)
            labels_joint.append(f"{material}_{motion}")
            # Create a binary label: "tap" vs "rub_push" (rub and push combined)
            if motion.lower() == "tap":
                labels_binary.append("tap")
            else:
                labels_binary.append("rub_push")
    
    X = np.array(features)
    return X, labels_material, labels_motion, labels_joint, labels_binary

def evaluate_classification(X, y, description=""):
    """
    Train a RandomForest classifier and print a concise classification summary.
    The summary includes the number of training and testing samples, along with 
    accuracy, precision, recall, and F1 score.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    print(f"--- {description} ---")
    print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print("-" * 60)

def plot_learning_curve(estimator, X, y, title="Learning Curve"):
    """
    Compute and plot the learning curve for an estimator.
    The learning curve shows the training loss and the cross-validated (validation) loss.
    Here, loss is defined as (1 - accuracy).
    """
    # Define a range for the number of training examples
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10), scoring='accuracy')
    
    # Calculate loss as (1 - accuracy)
    train_loss = 1 - np.mean(train_scores, axis=1)
    test_loss = 1 - np.mean(test_scores, axis=1)
    
    plt.figure()
    plt.plot(train_sizes, train_loss, 'o-', label="Training Loss")
    plt.plot(train_sizes, test_loss, 'o-', label="Validation Loss")
    plt.xlabel("Number of Training Examples")
    plt.ylabel("Loss (1 - Accuracy)")
    plt.title(title)
    plt.legend(loc="best")
    plt.grid(True)
    plt.show()

def evaluate_classification_with_learning_curve(X, y, description=""):
    """
    Train a RandomForest classifier, print evaluation metrics, and plot the learning curve
    to help inspect overfitting.
    """
    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)
    
    # Evaluate on training and testing sets
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)
    acc_train = accuracy_score(y_train, y_pred_train)
    acc_test = accuracy_score(y_test, y_pred_test)
    
    print(f"--- {description} ---")
    print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")
    print(f"Train Accuracy: {acc_train:.4f}")
    print(f"Test Accuracy:  {acc_test:.4f}")
    print("-" * 60)
    
    # Plot the learning curve (loss curve)
    #plot_learning_curve(clf, X, y, title=f"Learning Curve - {description}")

def main():
    csv_file = "experiment_data.csv"  # Update the path if needed
    window_size = 15  # Changeable window size (must be <= total samples per trial)
    
    # Load and process data with sliding window augmentation
    X, labels_material, labels_motion, labels_joint, labels_binary = load_and_process_data(csv_file, window_size)
    
    # Evaluate classification tasks:
    evaluate_classification(X, labels_material, description="Material Classification")
    evaluate_classification(X, labels_motion, description="Operation (Motion) Classification")
    evaluate_classification(X, labels_joint, description="Joint Material & Operation Classification")
    evaluate_classification(X, labels_binary, description="Binary Classification: Tap vs Rub/Push")
    
    # Additionally, inspect the learning curve for one of the tasks to check for overfitting.
    # For example, here we show the learning curve for the binary classification task.
    evaluate_classification_with_learning_curve(X, labels_material, description="Material Classification (Learning Curve)")
    evaluate_classification_with_learning_curve(X, labels_motion, description="Operation (Motion) Classification (Learning Curve)")
    evaluate_classification_with_learning_curve(X, labels_joint, description="Joint Material & Operation Classification (Learning Curve)")
    evaluate_classification_with_learning_curve(X, labels_binary, description="Binary Classification: Tap vs Rub/Push (Learning Curve)")

if __name__ == "__main__":
    main()
