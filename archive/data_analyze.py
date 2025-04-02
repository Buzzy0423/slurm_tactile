import pandas as pd
import numpy as np
import ast
from scipy.stats import linregress

# Load the CSV file (change the filename as needed)
df = pd.read_csv("experiment_data.csv")

# Parse the "reading" column which is stored as a string representation of a list
def parse_reading(reading_str):
    try:
        # Using ast.literal_eval to safely evaluate the string into a Python list
        return ast.literal_eval(reading_str)
    except Exception as e:
        print("Error parsing reading:", reading_str)
        return None

df["reading_parsed"] = df["reading"].apply(parse_reading)

# Group the data by material, motion, and trial.
grouped = df.groupby(["material", "motion", "trial"])

# Define a function to compute features from a trial’s time series data.
def compute_trial_features(trial_df):
    # Ensure the trial is sorted by timestamp
    trial_df = trial_df.sort_values("timestamp")
    # Create a 2D numpy array: shape (n_samples, 15)
    data = np.stack(trial_df["reading_parsed"].tolist(), axis=0)
    print("Data: ", data)  # Debugging line
    n_samples, n_channels = data.shape
    x = np.arange(n_samples)  # time indices (0 to 19)
    
    features = {}
    for ch in range(n_channels):
        channel_data = data[:, ch]
        # Basic time-series features:
        features[f"ch{ch}_mean"] = np.mean(channel_data)
        features[f"ch{ch}_std"] = np.std(channel_data)
        features[f"ch{ch}_min"] = np.min(channel_data)
        features[f"ch{ch}_max"] = np.max(channel_data)
        features[f"ch{ch}_ptp"] = np.ptp(channel_data)  # peak-to-peak difference
        
        # Compute the linear trend (slope) via linear regression:
        slope, _, _, _, _ = linregress(x, channel_data)
        features[f"ch{ch}_slope"] = slope
        
        # Optional: energy of the signal
        features[f"ch{ch}_energy"] = np.sum(channel_data**2) / n_samples
    return features

# Process each trial and store the extracted features along with condition labels.
trial_features = []
for (material, motion, trial), group in grouped:
    feats = compute_trial_features(group)
    feats["material"] = material
    feats["motion"] = motion
    feats["trial"] = trial
    trial_features.append(feats)

# Create a DataFrame from the trial features
features_df = pd.DataFrame(trial_features)

# Now, you can compute aggregated statistics on the extracted features.
# For example, to compute statistics for each material-motion combination:
mm_stats = features_df.groupby(["material", "motion"]).agg(["mean", "std"])

# Similarly, aggregated statistics can be computed for each motion (regardless of material)
motion_stats = features_df.groupby("motion").agg(["mean", "std"])

# And for each material (regardless of motion)
material_stats = features_df.groupby("material").agg(["mean", "std"])

# Finally, compute overall statistics across all trials.
overall_stats = features_df.agg(["mean", "std"])

# Display the results:
print("=== Material-Motion Level Statistics ===")
print(mm_stats)

print("\n=== Motion Level Statistics (aggregated over materials) ===")
print(motion_stats)

print("\n=== Material Level Statistics (aggregated over motions) ===")
print(material_stats)

print("\n=== Overall Statistics (all data) ===")
print(overall_stats)
