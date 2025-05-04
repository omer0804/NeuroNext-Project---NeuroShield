import os
import pandas as pd

# Define paths
base_dir = r"C:\Users\USER\Desktop\NeuroNext-Project---NeuroShield"
data_paths = {
    "processed": os.path.join(base_dir, "data", "processed"),
    "features": os.path.join(base_dir, "data", "features")
}

# Ensure features directory exists
os.makedirs(data_paths["features"], exist_ok=True)

# Function to extract features from physiological data
def extract_features():
    processed_files = [f for f in os.listdir(data_paths["processed"]) if f.startswith('processed_') and f.endswith('.csv')]
    features = []

    for file in processed_files:
        file_path = os.path.join(data_paths["processed"], file)
        df = pd.read_csv(file_path)

        # Example feature extraction: mean and standard deviation of a column (e.g., heart rate)
        if 'value' in df.columns:
            mean_value = df['value'].mean()
            std_value = df['value'].std()
            features.append({"id": file, "mean_value": mean_value, "std_value": std_value})

    # Save extracted features
    features_df = pd.DataFrame(features)
    features_df.to_csv(os.path.join(data_paths["features"], "physiological_features.csv"), index=False)

if __name__ == "__main__":
    extract_features()
    print("Feature extraction complete!")