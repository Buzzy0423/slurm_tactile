import time
import csv
import os
import numpy as np
from anyskin import AnySkinBase, AnySkinDummy  # Adjust the import path as necessary

# Configuration
SERIAL_PORT = "/dev/cu.usbmodem2101"
BAUDRATE = 115200
NUM_SAMPLES_PER_TRIAL = 20   # Number of sensor readings per trial
TRIALS_PER_CONDITION = 20    # Number of trials per condition

# Define the materials
materials = ["plastic", "wood", "metal"]
# labels = {"slide": 1, "static": 0}  # Binary classification labels
labels = {"slide": 1} 

# Flag to use dummy sensor
USE_DUMMY_SENSOR = False

def initialize_sensor():
    """Initializes and returns a sensor object."""
    if USE_DUMMY_SENSOR:
        print("Using dummy sensor for simulation.")
        sensor = AnySkinDummy(
            num_mags=5,
            port=SERIAL_PORT,
            baudrate=BAUDRATE,
            burst_mode=True,
            device_id=0,
            temp_filtered=True
        )
    else:
        sensor = AnySkinBase(
            num_mags=5,
            port=SERIAL_PORT,
        )
    return sensor

def get_baseline(sensor, num_baseline_samples=5):
    """
    Computes a baseline value by taking several sensor samples.
    Returns the mean sensor reading (as a NumPy array) for baseline correction.
    """
    baseline_samples = []
    print("Collecting baseline samples...")
    for _ in range(num_baseline_samples):
        data = sensor.get_sample()
        if len(data) == 2:
            _, reading = data
        elif len(data) == 3:
            _, _, reading = data
        else:
            raise ValueError("Unexpected sensor data format.")
        baseline_samples.append(np.array(reading))
        time.sleep(0.05)  # A short delay between baseline readings
    baseline = np.mean(baseline_samples, axis=0)
    print("Baseline computed.")
    return baseline

def collect_samples(sensor, num_samples, baseline):
    """
    Collects a specified number of baseline-corrected samples from the sensor.
    Each sensor reading is adjusted using the provided baseline.
    """
    samples = []
    count = 0
    while count < num_samples:
        data = sensor.get_sample()
        if len(data) == 2:
            timestamp, reading = data
        elif len(data) == 3:
            timestamp, _, reading = data
        else:
            raise ValueError("Unexpected sensor data format.")
        # Baseline correction: subtract the provided baseline from the reading
        adjusted_reading = np.array(reading) - baseline
        samples.append((timestamp, adjusted_reading.tolist()))
        count += 1
        time.sleep(0.1)
    return samples

def save_data_to_csv(filename, data):
    """Save the collected data to a CSV file."""
    os.makedirs(os.path.dirname(filename) or '.', exist_ok=True)
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["material", "label", "trial", "timestamp", "reading"])
        for entry in data:
            writer.writerow([entry["material"], entry["label"], entry["trial"], entry["timestamp"], entry["reading"]])
    print(f"Data saved to {filename}")

def main():
    sensor = initialize_sensor()
    collected_data = []
    
    print("Experiment Start: Binary Classification (Slide vs. Static) on Different Materials")
    print("Follow the instructions for each trial.")
    
    for material in materials:
        for label_name, label_value in labels.items():
            for trial in range(1, TRIALS_PER_CONDITION + 1):
                # First, update the baseline
                input(f"\nTrial {trial} for '{label_name}' contact on '{material}': Press Enter to update baseline...")
                baseline = get_baseline(sensor, num_baseline_samples=5)
                
                # Confirm baseline update and wait for the next key press to start collection
                input("Baseline updated. Press Enter to start data collection...")
                print(f"Collecting {NUM_SAMPLES_PER_TRIAL} samples for trial {trial} of {label_name} on {material}...")
                
                samples = collect_samples(sensor, NUM_SAMPLES_PER_TRIAL, baseline)
                
                for timestamp, reading in samples:
                    collected_data.append({
                        "material": material,
                        "label": label_value,
                        "trial": trial,
                        "timestamp": timestamp,
                        "reading": reading
                    })
                print(f"Completed trial {trial} for {label_name} on {material}.")
    
    output_filename = "./data/material_classification_data.csv"
    save_data_to_csv(output_filename, collected_data)
    print("Experiment complete. Dataset ready for analysis.")

if __name__ == "__main__":
    main()