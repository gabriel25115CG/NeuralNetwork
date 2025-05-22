import numpy as np

# Données brutes
raw_data = np.array([50, 100, 150, 200, 250], dtype=float)
raw_targets = np.array([150, 300, 450, 600, 750], dtype=float)

# Normalisation min-max
data_min, data_max = raw_data.min(), raw_data.max()
targets_min, targets_max = raw_targets.min(), raw_targets.max()

data = (raw_data - data_min) / (data_max - data_min)
targets = (raw_targets - targets_min) / (targets_max - targets_min)
