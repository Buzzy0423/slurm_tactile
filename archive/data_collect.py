import time
import csv
import os
from anyskin import AnySkinBase, AnySkinDummy  # Adjust the import path as necessary

# Configuration: Update these parameters as needed.
SERIAL_PORT = "/dev/cu.usbmodem101"         # get port by executing ls /dev/ | grep cu.usb
BAUDRATE = 115200
NUM_SAMPLES_PER_TRIAL = 20   # Number of sensor readings per trial
TRIALS_PER_CONDITION = 5     # Number of times to repeat each condition

# Define the materials and interaction types.
materials = ["wood", "metal", "fabric"]
motions = ["tap", "rub", "push"]

# Flag to use dummy sensor (if real sensor not available)
USE_DUMMY_SENSOR = False

def initialize_sensor():
    """
    Initializes and returns a sensor object.
    Use AnySkinDummy if USE_DUMMY_SENSOR is True.
    """
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
            # device_id=0,
            # temp_filtered=True,
            # burst_mode=True,
            # baudrate=BAUDRATE
        )
    return sensor

def collect_samples(sensor, num_samples):
    """
    Collect a specified number of samples from the sensor.
    
    Parameters
    ----------
    sensor : AnySkinBase or AnySkinDummy
        The sensor object from which to collect data.
    num_samples : int
        Number of samples to collect.
    
    Returns
    -------
    list of tuples:
        Each tuple contains (timestamp, sensor reading as list).
    """
    samples = []
    count = 0
    while count < num_samples:
        data = sensor.get_sample()
        # Handle both real and dummy sensor return types.
        if len(data) == 2:
            timestamp, reading = data
        elif len(data) == 3:
            timestamp, _, reading = data
        else:
            raise ValueError("Unexpected sensor data format.")
        samples.append((timestamp, list(reading)))
        count += 1
        time.sleep(0.1)  # Adjust delay as needed
    return samples

def save_data_to_csv(filename, data):
    """
    Save the collected data to a CSV file.
    
    Each row will contain: material, motion, trial number, timestamp, sensor_reading.
    
    Parameters
    ----------
    filename : str
        Output CSV file name.
    data : list of dicts
        Each dictionary contains keys: 'material', 'motion', 'trial', 'timestamp', 'reading'.
    """
    os.makedirs(os.path.dirname(filename) or '.', exist_ok=True)
    
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["material", "motion", "trial", "timestamp", "reading"])
        for entry in data:
            writer.writerow([entry["material"], entry["motion"], entry["trial"], entry["timestamp"], entry["reading"]])
    print(f"Data saved to {filename}")

def main():
    sensor = initialize_sensor()
    collected_data = []
    
    print("Experiment Start: Classification of Tap, Rub, and Push on Different Materials")
    print("NOTE: You will perform each action manually. The force is not precisely controlled, so repeating the experiment multiple times helps capture variability.")
    
    for material in materials:
        for motion in motions:
            for trial in range(1, TRIALS_PER_CONDITION + 1):
                input(f"\nTrial {trial} for '{motion}' on '{material}': Prepare and press Enter to start data collection...")
                print(f"Collecting {NUM_SAMPLES_PER_TRIAL} samples for trial {trial} of {motion} on {material}...")
                
                samples = collect_samples(sensor, NUM_SAMPLES_PER_TRIAL)
                
                for timestamp, reading in samples:
                    collected_data.append({
                        "material": material,
                        "motion": motion,
                        "trial": trial,
                        "timestamp": timestamp,
                        "reading": reading
                    })
                print(f"Completed trial {trial} for {motion} on {material}.")
    
    output_filename = "experiment_data.csv"
    save_data_to_csv(output_filename, collected_data)
    print("Experiment complete. You can now analyze the data further.")

if __name__ == "__main__":
    main()