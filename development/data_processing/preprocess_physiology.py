import os
import pandas as pd
import yaml

# Load settings from YAML
config_path = r"c:\Users\USER\Desktop\NeuroNext-Project---NeuroShield\config\settings.yaml"
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

data_paths = config['data_paths']

# Function to clean physiological data
def clean_physiological_data():
    for key in ['heart_rate', 'motion', 'steps']:
        folder_path = data_paths[key]
        files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

        for file in files:
            file_path = os.path.join(folder_path, file)
            df = pd.read_csv(file_path)

            # Remove rows with negative time values
            df = df[df['time'] >= 0]

            # Save processed data
            processed_path = os.path.join(data_paths['processed'], f"processed_{file}")
            df.to_csv(processed_path, index=False)

if __name__ == "__main__":
    clean_physiological_data()
    print("Physiological data preprocessing complete!")