#!/usr/bin/env python3
# Script to convert PyTorch nightmare detection model to CoreML format

import os
import torch
import torch.nn as nn
import numpy as np
import coremltools as ct

# Conv1D model with masking
class Conv1DClassifier(nn.Module):
    def __init__(self, input_channels, num_classes=2):
        super().__init__()
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=5, padding=2)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x, lengths):
        # Create a mask based on sequence lengths
        batch_size, seq_len, feature_dim = x.shape
        mask = torch.arange(seq_len, device=x.device)[None, :] < lengths[:, None]
        mask = mask.float().unsqueeze(1)  # (batch, 1, time)

        x = x.transpose(1, 2)  # (batch, features, time)
        x = self.relu1(self.conv1(x)) * mask
        x = self.relu2(self.conv2(x)) * mask

        # Masked average pooling
        masked_sum = torch.sum(x * mask, dim=2)
        masked_count = torch.sum(mask, dim=2) + 1e-6
        x = masked_sum / masked_count

        return self.fc(x)

def convert_to_coreml():
    # Set up paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "nightmare_detection_model.pth")
    output_path = os.path.join(base_dir, "ios_app", "NightmareDetectionModel.mlmodel")
    
    # Load PyTorch model
    input_channels = 16  # Defined in the original model
    model = Conv1DClassifier(input_channels=input_channels)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    
    # Create example inputs for tracing
    example_input = torch.rand(1, 120, input_channels)  # (batch_size, seq_len, features)
    example_lengths = torch.tensor([120], dtype=torch.int32)
    
    # Trace the model
    traced_model = torch.jit.trace(model, (example_input, example_lengths))
    
    # Define input and output descriptions
    input_desc = [
        ct.TensorType(name="features", shape=ct.Shape(shape=(1, ct.RangeDim(), input_channels))),
        ct.TensorType(name="lengths", shape=ct.Shape(shape=(1,)), dtype=np.int32)
    ]
    
    # Convert to CoreML
    mlmodel = ct.convert(
        traced_model, 
        inputs=input_desc,
        outputs=[ct.TensorType(name="output")],
        minimum_deployment_target=ct.target.iOS15,
        compute_precision=ct.precision.FLOAT16,
        convert_to="mlprogram"  # Use the new ML Program format
    )
    
    # Add metadata
    mlmodel.short_description = "NeuroShield Nightmare Detection Model"
    mlmodel.author = "NeuroNext"
    mlmodel.license = "Proprietary"
    mlmodel.version = "1.0"
    
    # Save the model
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    mlmodel.save(output_path)
    
    print(f"Model successfully converted and saved to {output_path}")

if __name__ == "__main__":
    convert_to_coreml()
