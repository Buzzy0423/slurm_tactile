#!/usr/bin/env python
import time
import numpy as np
import os
import sys
from datetime import datetime
from anyskin import AnySkinProcess
import argparse
from tensorflow.keras.models import load_model

def get_baseline(sensor_stream, num_samples=5):
    """
    Compute the baseline for sensor readings using a few samples.
    """
    baseline_data = sensor_stream.get_data(num_samples=num_samples)
    # Assume each sensor sample is a list where the first element is not part of the 15 sensor readings.
    baseline_data = np.array(baseline_data)[:, 1:]
    baseline = np.mean(baseline_data, axis=0)
    return baseline

def real_time_detection(port, model_path):
    # Load the sliding/static detection model (binary classification model with sigmoid output)
    model = load_model(model_path)
    
    # Start the sensor stream
    sensor_stream = AnySkinProcess(num_mags=5, port=port)
    sensor_stream.start()
    time.sleep(5.0)  # Allow the sensor stream to stabilize
    
    # Compute baseline for calibration
    baseline = get_baseline(sensor_stream, num_samples=5)
    print("Baseline computed:", baseline)
    
    recent_data = []   # sliding window for the last 10 sensor readings
    window_length = 5  # 10 samples => 1 second at 0.1s interval
    sensor_data_size = 15  # each sensor reading is a 15-dimensional vector
    
    sample_count = 0  # to track when 1 second has passed
    print("Starting real-time sliding/static detection. Press Ctrl+C to stop.")
    
    try:
        while True:
            # Get a single sensor sample; assume the first element is an identifier/timestamp.
            sample = sensor_stream.get_data(num_samples=1)[0][1:]
            # Calibrate the sample by subtracting the baseline.
            sample = np.array(sample) - baseline
            # Append to the sliding window.
            recent_data.append(sample)
            if len(recent_data) > window_length:
                recent_data.pop(0)
            sample_count += 1
            
            # Every 1 second (10 samples), perform prediction if we have a full window.
            if sample_count % 10 == 0 and len(recent_data) == window_length:
                # Reshape to match model input shape: (1, 10, 15)
                input_array = np.array(recent_data).reshape(1, window_length, sensor_data_size)
                # Model predicts a probability between 0 and 1.
                pred_prob = model.predict(input_array)[0][0]
                # Use a threshold of 0.5 to decide: >0.5 means "Slide", otherwise "Static".
                motion = "Slide" if pred_prob > 0.5 else "Static"
                print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Predicted Motion: {motion} (Probability: {pred_prob:.2f})")
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Real-time detection stopped by user.")
    finally:
        sensor_stream.pause_streaming()
        sensor_stream.join()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-time Sliding/Static Detection Using Trained LSTM Model")
    parser.add_argument("-p", "--port", type=str, default="/dev/cu.usbmodem101", help="Port to which the sensor is connected")
    parser.add_argument("-m", "--model", type=str, default="models/binary_classifier.keras", help="Path to the trained sliding/static detection model file")
    args = parser.parse_args()
    real_time_detection(args.port, args.model)