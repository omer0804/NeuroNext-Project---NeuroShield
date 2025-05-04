
# This script loads the configuration settings from a YAML file.
def load_settings(path = r"C:\Users\USER\Desktop\NeuroNext-Project---NeuroShield\config\settings.yaml"):
    import yaml
    with open(path, 'r') as file:
        return yaml.safe_load(file)