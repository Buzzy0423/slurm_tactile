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
    # Assume that each sensor sample is a list and that the first element is not part of the 15 sensor readings.
    baseline_data = np.array(baseline_data)[:, 1:]
    baseline = np.mean(baseline_data, axis=0)
    return baseline

def real_time_prediction(port, model_path):
    # Load the trained model (native Keras format)
    model = load_model(model_path)
    
    # Mapping from model's numeric output to material names
    material_mapping = {0: "plastic", 1: "wood", 2: "metal"}
    
    # Start sensor stream
    sensor_stream = AnySkinProcess(num_mags=5, port=port)
    sensor_stream.start()
    time.sleep(5.0)  # Let the sensor stream stabilize
    
    # Get baseline from the sensor data for calibration
    baseline = get_baseline(sensor_stream, num_samples=5)
    print("Baseline computed:", baseline)
    
    recent_data = []  # to store the last 10 sensor readings
    window_length = 5   # last 10 data points (1 sec at 0.1s interval)
    sensor_data_size = 15  # each sensor reading is a 15-dimensional vector
    
    sample_count = 0  # counter to control prediction frequency
    print("Starting real-time material prediction. Press Ctrl+C to stop.")
    try:
        while True:
            # Get a single sensor data sample (skip the first element if it's an identifier)
            sample = sensor_stream.get_data(num_samples=1)[0][1:]
            # Subtract baseline for calibration
            sample = np.array(sample) - baseline
            # Append the new sample to the sliding window
            recent_data.append(sample)
            # Keep only the last 10 samples
            if len(recent_data) > window_length:
                recent_data.pop(0)
            sample_count += 1
            
            # Every 1 second (i.e., every 10 samples), perform prediction if window is full.
            if sample_count % 10 == 0 and len(recent_data) == window_length:
                # Reshape to match model input: (1, 10, 15)
                input_array = np.array(recent_data).reshape(1, window_length, sensor_data_size)
                pred_prob = model.predict(input_array)
                pred_class = np.argmax(pred_prob, axis=1)[0]
                material = material_mapping.get(pred_class, "Unknown")
                print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Predicted Material: {material} - Probability Distribution: {pred_prob[0]} (Class Index: {pred_class})")
            # Wait 0.1 seconds before collecting next sample
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Real-time prediction stopped by user.")
    finally:
        sensor_stream.pause_streaming()
        sensor_stream.join()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-time Material Prediction Using Trained LSTM Model")
    parser.add_argument("-p", "--port", type=str, default="/dev/cu.usbmodem2101", help="Port to which the sensor is connected")
    parser.add_argument("-m", "--model", type=str, default="models/material_classifier.keras", help="Path to the trained model file")
    args = parser.parse_args()
    real_time_prediction(args.port, args.model)