"""
check_train_data_maps.py

Utility script to inspect the shape and contents of the synthetic TRACE-compatible 
`train_encoder_data_maps.npy` file. This file is used to map training samples to 
encoder metadata such as group indices or domain labels.

Expected file format:
- A NumPy array of shape (N, 2), where:
  - Column 0: sample index
  - Column 1: group label or encoder fold index (typically 0 in synthetic data)
"""

import numpy as np

# Load the training encoder data map from the synthetic HiRID dataset
maps = np.load("DONTCOMMITdata/hirid_numpy/train_encoder_data_maps.npy")

# Print the shape of the loaded data (should be [num_samples, 2])
print("Shape of train_encoder_data_maps:", maps.shape)

# Print the first 5 rows to verify contents
print("First 5 entries:\n", maps[:5])

