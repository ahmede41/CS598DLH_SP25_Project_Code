"""
check_train.py

Utility script to inspect the contents and shape of the `train.pt` file generated 
as part of the synthetic HiRID dataset preparation for TRACE. This file contains 
the normalized and preprocessed input tensors used to train the encoder.

Expected:
- A PyTorch tensor of shape (N, T, C), typically:
  - N: number of samples
  - T: time steps (e.g., 60)
  - C: channels/features (e.g., 36 if repeated from 6)

This script verifies:
- Type of object loaded from disk
- Shape of the tensor or first entry if it's a list
"""

import torch

# Load the training dataset tensor from the synthetic output directory
train_encoder_data = torch.load("synthetic_hirid_dataset/train.pt", weights_only=False)

# Print the type of loaded object (should be torch.Tensor)
print("Loaded object type:", type(train_encoder_data))

# Print the shape of the first item if it's a list, else the full tensor shape
if isinstance(train_encoder_data, list):
    print("First sample shape:", train_encoder_data[0].shape)
else:
    print("Full tensor shape:", train_encoder_data.shape)

