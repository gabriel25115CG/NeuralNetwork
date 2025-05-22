import numpy as np

data = np.array([50, 100, 150, 200, 250], dtype=float)
targets = np.array([150, 300, 450, 600, 750], dtype=float)

data_min, data_max = data.min(), data.max()
targets_min, targets_max = targets.min(), targets.max()
