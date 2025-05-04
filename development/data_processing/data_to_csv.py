## imports

import os
import sys

# Dynamically add the utils directory to the Python path
current_dir = os.path.dirname(__file__)
utils_dir = os.path.abspath(os.path.join(current_dir, '../utils'))
sys.path.append(utils_dir)

# Dynamically add the config directory to the Python path
config_dir = os.path.abspath(os.path.join(current_dir, '../config'))
sys.path.append(config_dir)

from config import load_settings  # noqa: E402

# function to change data files from txt to csv (one time run), do not run if files are already in csv format
config = load_settings()

def txt_data_to_csv():
    # Validate the presence of required keys in the config dictionary
    try:
        data_dirs = config["data_paths"]
    except KeyError as e:
        print(f"Configuration error: Missing key {e}")
        return

    for dir_key, dir_path in data_dirs.items():
        # Get all files in the directory
        if not os.path.exists(dir_path):
            print(f"Directory not found: {dir_path}")
            continue

        print(f"Processing directory: {dir_path}")
        for file in os.listdir(dir_path):
            if file.endswith('.txt'):
                file_path = os.path.join(dir_path, file)
                # Read the file
                with open(file_path, 'r') as f:
                    lines = f.readlines()
                    # Each space in the txt file is a new column
                    data = [line.split() for line in lines]
                    # Create a new file with the same name but with .csv extension
                    new_file = file_path.replace('.txt', '.csv')
                    with open(new_file, 'w') as f:
                        for line in data:
                            # Join the columns with a comma
                            f.write(','.join(line) + '\n')
                # Delete the original .txt file after conversion
                os.remove(file_path)

if __name__ == "__main__":
    # run the function to convert txt files to csv
    txt_data_to_csv()
    print("Conversion complete!")


