import os
import pandas as pd
import yaml

# Load settings from YAML
config_path = r"c:\Users\USER\Desktop\NeuroNext-Project---NeuroShield\config\settings.yaml"
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

data_paths = config['data_paths']

# Function to preprocess labels and extract nightmare feature
def preprocess_labels():
    label_files = [f for f in os.listdir(data_paths['lables']) if f.endswith('.csv')]
    nightmare_features = []

    for file in label_files:
        file_path = os.path.join(data_paths['lables'], file)
        df = pd.read_csv(file_path)

        # Remove rows with label -1
        df = df[df['label'] != -1]

        # Extract nightmare feature
        for i in range(len(df) - 1):
            if df.iloc[i]['label'] == 5 and df.iloc[i + 1]['label'] == 0:
                nightmare_features.append({"id": file, "nightmare_detected": 1})
                break
        else:
            nightmare_features.append({"id": file, "nightmare_detected": 0})

        # Save processed labels
        processed_path = os.path.join(data_paths['processed'], f"processed_{file}")
        df.to_csv(processed_path, index=False)

    # Save nightmare features table
    features_df = pd.DataFrame(nightmare_features)
    features_df.to_csv(os.path.join(data_paths['processed'], "nightmare_features.csv"), index=False)

if __name__ == "__main__":
    preprocess_labels()
    print("Labels preprocessing and nightmare feature extraction complete!")